"""Train / evaluate the multi-task ST-GNN and its single-task baselines.

Examples
  python train.py --data synthetic --epochs 3 --out results/smoke.json
  python train.py --data metr-la --data_dir data/METR-LA --mode multi --out results/multi_fixed.json
"""
import argparse
import copy
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from data import IGNORE, prepare
from metrics import cls_metrics, reg_by_level, reg_metrics
from model import MTSTGNN, count_params

AMP = {"on": False}  # set by --amp (CUDA only): encoder in float16, heads and losses in float32


def fwd(model, x):
    with torch.autocast("cuda", dtype=torch.float16, enabled=AMP["on"]):
        return model(x)


# --------------------------------------------------------------------------- data
class Windows(Dataset):
    def __init__(self, d, split, P, Q):
        self.f, self.s, self.l = d["feats"], d["speed"], d["labels"]
        self.starts, self.P, self.Q = d["starts"][split], P, Q

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, i):
        s = self.starts[i]
        return (torch.from_numpy(self.f[s:s + self.P]),
                torch.from_numpy(self.s[s + self.P:s + self.P + self.Q]),
                torch.from_numpy(self.l[s + self.P:s + self.P + self.Q]))


# --------------------------------------------------------------------------- losses
def masked_mae(pred_n, true_raw, mean, std):
    """MAE on the normalised scale; zeros in true_raw are missing and ignored."""
    mask = (true_raw > 0).float()
    return (torch.abs(pred_n - (true_raw - mean) / std) * mask).sum() / mask.sum().clamp(min=1)


def ce_loss(logits, labels, weight):
    if (labels != IGNORE).sum() == 0:
        return logits.sum() * 0.0
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1),
                           weight=weight, ignore_index=IGNORE)


def task_losses(out, yr, yc, mean, std, cw):
    reg, cls = out
    lr = masked_mae(reg, yr, mean, std) if reg is not None else None
    lc = ce_loss(cls, yc, cw) if cls is not None else None
    return lr, lc


def weigh(lr, lc, args, log_vars):
    """Return the list of (weighted) task losses that are summed for training."""
    if lr is None:
        return [lc]
    if lc is None:
        return [lr]
    if args.weighting == "uncertainty":  # Kendall et al.: exp(-s) * L + s
        return [torch.exp(-log_vars[0]) * lr + log_vars[0],
                torch.exp(-log_vars[1]) * lc + log_vars[1]]
    return [args.alpha * lr, args.beta * lc]


# --------------------------------------------------------------------------- gradient tools
def _flat(grads, params):
    return torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1)
                      for g, p in zip(grads, params)])


def shared_grad_cos(enc_params, losses, scale=1.0):
    """Cosine similarity between the two task gradients on the shared encoder
    (scale = static loss scale used with --amp; the cosine does not depend on it)."""
    g = [_flat(torch.autograd.grad(l * scale, enc_params, retain_graph=True, allow_unused=True), enc_params)
         for l in losses]
    return F.cosine_similarity(g[0], g[1], dim=0).item()


def pcgrad_backward(enc_params, other_params, losses, scale=1.0):
    """PCGrad (Yu et al., 2020) on the shared encoder; heads get their own gradients.
    Returns the cosine similarity before projection. With --amp the losses are multiplied by `scale`
    before autograd and the gradients divided by it afterwards (so the projection sees true gradients)."""
    g = [_flat(torch.autograd.grad(l * scale, enc_params, retain_graph=True, allow_unused=True), enc_params) / scale
         for l in losses]
    cos = F.cosine_similarity(g[0], g[1], dim=0).item()
    proj = [x.clone() for x in g]
    for i in range(len(proj)):
        others = [j for j in range(len(g)) if j != i]
        random.shuffle(others)
        for j in others:
            dot = torch.dot(proj[i], g[j])
            if dot < 0:
                proj[i] = proj[i] - dot / (g[j].norm() ** 2 + 1e-12) * g[j]
    merged, off = sum(proj), 0
    for p in enc_params:
        n = p.numel()
        p.grad = merged[off:off + n].view_as(p).clone()
        off += n
    total = sum(losses) * scale
    for p, gr in zip(other_params, torch.autograd.grad(total, other_params, allow_unused=True)):
        p.grad = None if gr is None else gr / scale
    return cos


# --------------------------------------------------------------------------- evaluation
@torch.no_grad()
def evaluate(model, loader, device, mean, std, cw, collect=False):
    model.eval()
    tot, n = 0.0, 0
    R, TR, L, TL, PC, YL = [], [], [], [], [], []
    for x, yr, yc in loader:
        x, yr, yc = x.to(device), yr.to(device), yc.to(device)
        reg, cls = fwd(model, x)
        lr, lc = task_losses((reg, cls), yr, yc, mean, std, cw)
        tot += sum(l.item() for l in (lr, lc) if l is not None)  # unweighted sum, used for model selection
        n += 1
        if collect:
            YL.append(yc.cpu().numpy())
            if reg is not None:
                R.append((reg * std + mean).cpu().numpy()); TR.append(yr.cpu().numpy())
            if cls is not None:
                pr = cls.softmax(-1)
                L.append(pr.argmax(-1).cpu().numpy()); PC.append(pr[..., -1].cpu().numpy())
                TL.append(yc.cpu().numpy())
    res = {"loss": tot / max(n, 1)}
    if collect:
        if R:
            res["reg"] = reg_metrics(np.concatenate(R), np.concatenate(TR))
            res["reg_by_level"] = reg_by_level(np.concatenate(R), np.concatenate(TR), np.concatenate(YL))
        if L:
            res["cls"] = cls_metrics(np.concatenate(L), np.concatenate(TL), np.concatenate(PC))
    return res


@torch.no_grad()
def latency_ms(model, x, device, reps=50):
    model.eval()
    x = x.to(device)
    for _ in range(10):
        model(x)
    ts = []
    for _ in range(reps):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000)
    return float(np.median(ts))


@torch.no_grad()
def dump_arrays(model, loader, device, mean, std):
    """Sequential (unshuffled) predictions, needed by eval_warning.py (windows are consecutive in time)."""
    model.eval()
    R, TR, LG, TL = [], [], [], []
    for x, yr, yc in loader:
        reg, cls = fwd(model, x.to(device))
        TL.append(yc.numpy().astype(np.int8))  # labels are always saved (needed for event analysis)
        if reg is not None:
            R.append((reg * std + mean).cpu().numpy()); TR.append(yr.numpy())
        if cls is not None:
            LG.append(cls.cpu().numpy().astype(np.float16))
    out = {"labels": np.concatenate(TL)}
    if R:
        out["reg_pred"], out["reg_true"] = np.concatenate(R).astype(np.float32), np.concatenate(TR)
    if LG:
        out["logits"] = np.concatenate(LG)
    return out


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="synthetic", choices=["synthetic", "metr-la", "pems-bay"])
    ap.add_argument("--data_dir", default=".")
    ap.add_argument("--mode", default="multi", choices=["multi", "reg", "cls"])
    ap.add_argument("--weighting", default="fixed", choices=["fixed", "uncertainty"])
    ap.add_argument("--alpha", type=float, default=1.0, help="weight of the regression loss")
    ap.add_argument("--beta", type=float, default=1.0, help="weight of the classification loss")
    ap.add_argument("--pcgrad", action="store_true")
    ap.add_argument("--class_weight", action="store_true", help="sqrt inverse-frequency CE weights")
    ap.add_argument("--persist_k", type=int, default=3, help="label persistence (1 = plain threshold)")
    ap.add_argument("--t_warn", type=float, default=0.70, help="speed-ratio threshold between level 0 and 1")
    ap.add_argument("--t_crit", type=float, default=0.40, help="speed-ratio threshold between level 1 and 2")
    ap.add_argument("--P", type=int, default=12)
    ap.add_argument("--Q", type=int, default=12)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--no_adaptive", action="store_true")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--log_grad_cos", action="store_true")
    ap.add_argument("--save_preds", action="store_true", help="save val/test predictions (.npz) for eval_warning.py")
    ap.add_argument("--max_batches", type=int, default=0, help="limit batches/epoch (smoke tests)")
    ap.add_argument("--amp", action="store_true", help="float16 autocast for the encoder (CUDA), static+backoff loss scaling")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/run.json")
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    AMP["on"] = bool(args.amp and device.type == "cuda")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    tasks = ("reg", "cls") if args.mode == "multi" else (args.mode,)

    d = prepare(args.data, args.data_dir, args.P, args.Q, args.persist_k, args.seed, args.t_warn, args.t_crit)
    mean, std = d["mean"], d["std"]
    cnt = d["class_counts"]
    print(f"nodes={d['n_nodes']}  train class counts (safe/warn/disrupt)={cnt.astype(int).tolist()}")
    cw = None
    if args.class_weight:
        w = np.sqrt(cnt.sum() / (3 * np.maximum(cnt, 1)))
        cw = torch.tensor(w / w.mean(), dtype=torch.float32, device=device)

    loaders = {s: DataLoader(Windows(d, s, args.P, args.Q), batch_size=args.batch_size,
                             shuffle=(s == "train"), num_workers=0)
               for s in ("train", "val", "test")}

    model = MTSTGNN(d["n_feats"], d["n_nodes"], d["adj"], args.Q, 3, tasks,
                    args.hidden, args.layers, adaptive=not args.no_adaptive).to(device)
    log_vars = nn.Parameter(torch.zeros(2, device=device))
    use_uw = args.mode == "multi" and args.weighting == "uncertainty"
    params = list(model.parameters()) + ([log_vars] if use_uw else [])
    opt = torch.optim.Adam(params, lr=args.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3)
    enc_params = list(model.encoder.parameters())
    other_params = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    if use_uw:
        other_params.append(log_vars)
    two_tasks = args.mode == "multi"
    n_params = count_params(model)
    print(f"mode={args.mode} weighting={args.weighting} pcgrad={args.pcgrad} params={n_params:,} device={device}")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    best, best_state, bad = float("inf"), None, 0
    hist, cos_log, epoch_times = [], [], []
    loss_scale, n_skip = {"s": 1024.0 if AMP["on"] else 1.0, "good": 0}, 0
    for ep in range(1, args.epochs + 1):
        model.train()
        t0, run, nb = time.time(), 0.0, 0
        for b, (x, yr, yc) in enumerate(loaders["train"]):
            if args.max_batches and b >= args.max_batches:
                break
            x, yr, yc = x.to(device), yr.to(device), yc.to(device)
            lr_, lc_ = task_losses(fwd(model, x), yr, yc, mean, std, cw)
            losses = weigh(lr_, lc_, args, log_vars)
            opt.zero_grad()
            S = loss_scale["s"]
            if two_tasks and args.pcgrad:
                cos = pcgrad_backward(enc_params, other_params, losses, S)
                if math.isfinite(cos):
                    cos_log.append((ep, b, cos))
            else:
                if two_tasks and args.log_grad_cos and b % 20 == 0:
                    cos = shared_grad_cos(enc_params, losses, S)
                    if math.isfinite(cos):
                        cos_log.append((ep, b, cos))
                (sum(losses) * S).backward()
                if S != 1.0:
                    for p in params:
                        if p.grad is not None:
                            p.grad.div_(S)
            gn = nn.utils.clip_grad_norm_(params, 5.0)
            if AMP["on"] and not torch.isfinite(gn):  # float16 overflow: skip this step, halve the loss scale
                loss_scale["s"], loss_scale["good"] = max(S / 2, 1.0), 0
                n_skip += 1
                continue
            opt.step()
            loss_scale["good"] += 1
            if AMP["on"] and loss_scale["good"] % 500 == 0:
                loss_scale["s"] = min(S * 2, 2.0 ** 15)
            run += sum(l.item() for l in losses); nb += 1
        epoch_times.append(time.time() - t0)
        val = evaluate(model, loaders["val"], device, mean, std, cw)["loss"]
        sched.step(val)
        hist.append(dict(epoch=ep, train=run / max(nb, 1), val=val))
        print(f"epoch {ep:3d}  train {run / max(nb, 1):.4f}  val {val:.4f}  ({epoch_times[-1]:.1f}s)")
        if val < best - 1e-5:
            best, best_state, bad = val, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= args.patience:
                print("early stop"); break
    if best_state is not None:
        model.load_state_dict(best_state)

    test = evaluate(model, loaders["test"], device, mean, std, cw, collect=True)
    xb = next(iter(loaders["test"]))[0]
    eff = dict(params=n_params, train_s_per_epoch=float(np.mean(epoch_times)),
               latency_ms_batch1=latency_ms(model, xb[:1], device),
               latency_ms_batch=latency_ms(model, xb, device), batch=int(xb.shape[0]),
               peak_gpu_mb=(torch.cuda.max_memory_allocated() / 2 ** 20 if device.type == "cuda" else None))

    if "reg" in test:
        a = test["reg"]["avg"]
        print(f"TEST reg  MAE {a['mae']:.3f}  RMSE {a['rmse']:.3f}  MAPE {a['mape']:.2f}%")
    if "cls" in test:
        a = test["cls"]["avg"]
        print(f"TEST cls  acc {a['acc']:.3f}  macroF1 {a['macro_f1']:.3f}  "
              f"recall(L2) {a['recall'][2]:.3f}  PR-AUC(L2) {a.get('pr_auc_level2', float('nan')):.3f}")
    eff["amp"], eff["amp_skipped_steps"] = AMP["on"], n_skip
    print("efficiency:", json.dumps(eff))

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    if args.save_preds:
        arrs = {f"{sp}_{k}": v for sp in ("val", "test")
                for k, v in dump_arrays(model, loaders[sp], device, mean, std).items()}
        arrs["v_free"] = d["v_free"]
        np.savez_compressed(out.with_suffix(".npz"), **arrs)
        print("saved predictions", out.with_suffix(".npz"))
    out.write_text(json.dumps(dict(args=vars(args), test=test, efficiency=eff, history=hist,
                                   grad_cos=cos_log, class_counts=cnt.tolist()), indent=2))
    print("saved", out)


if __name__ == "__main__":
    main()
