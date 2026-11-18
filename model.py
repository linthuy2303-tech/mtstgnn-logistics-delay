"""Multi-task spatio-temporal GNN: shared encoder + task-specific heads.

Encoder (shared):  1x1 conv -> [gated causal TCN -> graph conv (+ adaptive adjacency)] x L
Heads (separate):  regression head -> Q future speeds per node
                   classification head -> Q x 3 risk-level logits per node
`tasks` selects which heads exist, so single-task baselines have no unused parameters.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_supports(adj):
    """Forward / backward random-walk normalised transition matrices."""
    def rw(a):
        d = a.sum(1, keepdims=True)
        d[d == 0] = 1.0
        return a / d
    adj = np.asarray(adj, dtype=np.float32)
    return [torch.from_numpy(rw(adj)), torch.from_numpy(rw(adj.T.copy()))]


class GatedTCN(nn.Module):
    def __init__(self, d, k=3, dilation=1):
        super().__init__()
        self.pad = (k - 1) * dilation
        self.f = nn.Conv2d(d, d, (1, k), dilation=(1, dilation))
        self.g = nn.Conv2d(d, d, (1, k), dilation=(1, dilation))

    def forward(self, x):  # (B, D, N, T) -> same shape, causal
        x = F.pad(x, (self.pad, 0))
        return torch.tanh(self.f(x)) * torch.sigmoid(self.g(x))


class GraphConv(nn.Module):
    def __init__(self, d, n_supports, order=2, dropout=0.3):
        super().__init__()
        self.order = order
        self.mlp = nn.Conv2d(d * (1 + n_supports * order), d, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, supports):  # x: (B, D, N, T)
        out = [x]
        for A in supports:
            h = x
            for _ in range(self.order):
                h = torch.einsum("mn,bcnt->bcmt", A, h)
                out.append(h)
        return self.drop(self.mlp(torch.cat(out, dim=1)))


class Encoder(nn.Module):
    def __init__(self, c_in, n_nodes, supports, d=32, layers=3, k=3,
                 dropout=0.3, adaptive=True, emb=10):
        super().__init__()
        self.n_fixed = len(supports)
        for i, s in enumerate(supports):
            self.register_buffer(f"support_{i}", s)
        self.adaptive = adaptive
        if adaptive:
            self.e1 = nn.Parameter(torch.randn(n_nodes, emb) * 0.1)
            self.e2 = nn.Parameter(torch.randn(n_nodes, emb) * 0.1)
        n_sup = self.n_fixed + (1 if adaptive else 0)
        self.inp = nn.Conv2d(c_in, d, 1)  # 1x1 conv: channel projection / compression
        self.tcn = nn.ModuleList([GatedTCN(d, k, 2 ** i) for i in range(layers)])
        self.gcn = nn.ModuleList([GraphConv(d, n_sup, 2, dropout) for _ in range(layers)])
        self.bn = nn.ModuleList([nn.BatchNorm2d(d) for _ in range(layers)])
        self.skip = nn.ModuleList([nn.Conv2d(d, d, 1) for _ in range(layers)])
        self.out_dim = d

    def forward(self, x):  # x: (B, P, N, C)
        x = self.inp(x.permute(0, 3, 2, 1))  # (B, D, N, P)
        sup = [getattr(self, f"support_{i}") for i in range(self.n_fixed)]
        if self.adaptive:
            sup.append(F.softmax(F.relu(self.e1 @ self.e2.T), dim=1))
        skip = 0
        for tcn, gcn, bn, sk in zip(self.tcn, self.gcn, self.bn, self.skip):
            res = x
            x = gcn(tcn(x), sup)
            skip = skip + sk(x)[..., -1:]  # keep the last time step
            x = bn(x + res)
        return F.relu(skip).squeeze(-1).permute(0, 2, 1)  # (B, N, D)


class MTSTGNN(nn.Module):
    def __init__(self, c_in, n_nodes, adj, horizon=12, n_cls=3, tasks=("reg", "cls"),
                 d=32, layers=3, dropout=0.3, adaptive=True):
        super().__init__()
        self.horizon, self.n_cls, self.tasks = horizon, n_cls, tuple(tasks)
        self.encoder = Encoder(c_in, n_nodes, build_supports(adj), d, layers,
                               dropout=dropout, adaptive=adaptive)
        if "reg" in self.tasks:
            self.reg_head = nn.Sequential(nn.Linear(d, 2 * d), nn.ReLU(), nn.Linear(2 * d, horizon))
        if "cls" in self.tasks:
            self.cls_head = nn.Sequential(nn.Linear(d, 2 * d), nn.ReLU(),
                                          nn.Linear(2 * d, horizon * n_cls))

    def forward(self, x):
        h = self.encoder(x)  # (B, N, D); may be float16 when called under autocast (train.py --amp)
        with torch.autocast(device_type=h.device.type, enabled=False):  # heads and losses always in float32
            h = h.float()
            B, N, _ = h.shape
            reg = self.reg_head(h).permute(0, 2, 1) if "reg" in self.tasks else None        # (B, Q, N)
            cls = (self.cls_head(h).view(B, N, self.horizon, self.n_cls).permute(0, 2, 1, 3)  # (B, Q, N, 3)
                   if "cls" in self.tasks else None)
        return reg, cls


def count_params(model):
    return sum(p.numel() for p in model.parameters())
