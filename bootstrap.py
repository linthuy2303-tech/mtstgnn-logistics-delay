"""Block bootstrap over test days for paired differences between variants (numpy only).

The test windows are consecutive in time, so they are cut into blocks of 288 windows (one day of 5-minute steps).
Each replicate draws blocks with replacement (the SAME blocks for both variants of a comparison) and recomputes
  * speed MAE (pooled over the 12 steps, zeros in the target are missing),
  * macro-F1 (pooled over the 12 steps, classes present in the replicate),
  * average precision (AP) of risk level 2 (pooled), computed from 2000-bin histograms of the level-2 probability
    so that all replicates are evaluated at once (the binned AP differs from the exact AP by a few 1e-3 at most).
Only seed 0 is used, because predictions (.npz) are kept for seed 0 only. This measures test-set sampling
uncertainty (23-24 test days); it does not include training randomness, which is covered by the seeds.

Usage:
    python bootstrap.py --dir results --best mtf_b2 --out results/bootstrap_s0.json
    python bootstrap.py --selftest
"""
import argparse
import json
from pathlib import Path

import numpy as np

IGNORE = -100
BLOCK = 288
NBINS = 2000


def softmax_level2(logits):
    z = logits.astype(np.float32)
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e[..., 2] / e.sum(-1)


def blocks_of(n, size=BLOCK):
    return [(i, min(i + size, n)) for i in range(0, n, size)]


class Variant:
    """Per-block sufficient statistics of one run's test predictions."""

    def __init__(self, npz_path, nbins=NBINS):
        d = np.load(npz_path)
        self.has_reg = "test_reg_pred" in d
        self.has_cls = "test_logits" in d
        labels = d["test_labels"].astype(np.int64)
        n = labels.shape[0]
        self.bl = blocks_of(n)
        nb = len(self.bl)
        if self.has_reg:
            pred, true = d["test_reg_pred"], d["test_reg_true"]
            m = true > 0
            ae = np.where(m, np.abs(pred - true), 0.0)
            self.ae = np.array([ae[a:b].sum(dtype=np.float64) for a, b in self.bl])
            self.nobs = np.array([m[a:b].sum() for a, b in self.bl], dtype=np.float64)
        if self.has_cls:
            logits = d["test_logits"]
            pred_c = logits.argmax(-1)
            p2 = softmax_level2(logits)
            valid = labels != IGNORE
            cm = np.zeros((nb, 3, 3))
            for k, (a, b) in enumerate(self.bl):
                v = valid[a:b]
                t, p = labels[a:b][v], pred_c[a:b][v]
                cm[k] = np.bincount(t * 3 + p, minlength=9).reshape(3, 3)
            self.cm = cm                                     # rows = true, cols = predicted
            sample = p2[valid]
            if sample.size > 2_000_000:
                sample = np.random.default_rng(0).choice(sample, 2_000_000, replace=False)
            edges = np.unique(np.quantile(sample, np.linspace(0, 1, nbins + 1)[1:-1]))
            self.nb_eff = len(edges) + 1
            hp, hn = np.zeros((nb, self.nb_eff)), np.zeros((nb, self.nb_eff))
            for k, (a, b) in enumerate(self.bl):
                v = valid[a:b]
                idx = np.searchsorted(edges, p2[a:b][v], side="right")
                pos = labels[a:b][v] == 2
                hp[k] = np.bincount(idx[pos], minlength=self.nb_eff)
                hn[k] = np.bincount(idx[~pos], minlength=self.nb_eff)
            self.hp, self.hn = hp[:, ::-1], hn[:, ::-1]      # descending score order

    # counts: (R, nb) number of times each block is drawn (R = 1 row of ones gives the full sample)
    def mae(self, counts):
        return (counts @ self.ae) / (counts @ self.nobs)

    def macro_f1(self, counts):
        cm = np.einsum("rb,bij->rij", counts, self.cm)
        tp = np.einsum("rii->ri", cm)
        support, predn = cm.sum(2), cm.sum(1)
        rec = tp / np.maximum(support, 1)
        prec = tp / np.maximum(predn, 1)
        f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
        present = support > 0
        return (f1 * present).sum(1) / np.maximum(present.sum(1), 1)

    def ap2(self, counts):
        hp, hn = counts @ self.hp, counts @ self.hn
        tp, fp = np.cumsum(hp, 1), np.cumsum(hn, 1)
        prec = tp / np.maximum(tp + fp, 1e-12)
        total = np.maximum(tp[:, -1:], 1e-12)
        return ((hp / total) * prec).sum(1)


METRICS = {"mae": ("has_reg", "mae"), "f1": ("has_cls", "macro_f1"), "ap2": ("has_cls", "ap2")}


def compare(va, vb, metric, reps, rng):
    flag, fn = METRICS[metric]
    if not (getattr(va, flag) and getattr(vb, flag)):
        return None
    nb = len(va.bl)
    assert nb == len(vb.bl), "variants have different numbers of test windows"
    draws = rng.integers(0, nb, size=(reps, nb))
    counts = np.stack([np.bincount(r, minlength=nb) for r in draws]).astype(np.float64)
    full = np.ones((1, nb))
    fa, fb = getattr(va, fn), getattr(vb, fn)
    point = float((fa(full) - fb(full))[0])
    diff = fa(counts) - fb(counts)
    lo, hi = np.percentile(diff, [2.5, 97.5])
    return dict(metric=metric, a_full=float(fa(full)[0]), b_full=float(fb(full)[0]), diff=point,
                ci_lo=float(lo), ci_hi=float(hi), share_a_lower=float((diff < 0).mean()), reps=reps, blocks=nb)


def run(d, best, seed=0, reps=2000, out=None):
    d = Path(d)
    names = {"R": "reg", "C": "cls", "MT-F (best β)": best, "MT-U": "mtu", "MT-P": "mtp"}
    V = {}
    for k, var in names.items():
        p = d / f"{var}_s{seed}.npz"
        if var and p.exists():
            V[k] = Variant(p)
    rng = np.random.default_rng(12345)
    res = []
    for mt in ("MT-F (best β)", "MT-U", "MT-P"):
        if mt not in V:
            continue
        for ref, met in [("R", "mae"), ("C", "ap2"), ("C", "f1")]:
            if ref in V:
                r = compare(V[mt], V[ref], met, reps, rng)
                if r:
                    res.append(dict(a=mt, b=ref, **r))
    for mt in ("MT-U", "MT-P"):
        if mt in V and "MT-F (best β)" in V:
            for met in ("mae", "ap2", "f1"):
                r = compare(V[mt], V["MT-F (best β)"], met, reps, rng)
                if r:
                    res.append(dict(a=mt, b="MT-F (best β)", **r))
    for r in res:
        print(f"{r['a']:>14} - {r['b']:<14} {r['metric']:>4}: {r['diff']:+.4f}  95% CI [{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}]"
              f"  (A lower in {100 * r['share_a_lower']:.0f}% of replicates; blocks {r['blocks']})")
    if out:
        Path(out).write_text(json.dumps(dict(seed=seed, block=BLOCK, bins=NBINS, comparisons=res), indent=1))
    return res


# --------------------------------------------------------------------------- self-test on synthetic predictions
def selftest():
    import tempfile
    from metrics import average_precision, cls_metrics, reg_metrics
    rng = np.random.default_rng(0)
    B, Q, N = 1000, 12, 8
    true = rng.uniform(5, 70, (B, Q, N)).astype(np.float32)
    true[rng.random(true.shape) < 0.02] = 0.0
    labels = rng.choice([0, 1, 2], size=(B, Q, N), p=[0.9, 0.07, 0.03]).astype(np.int8)
    labels[rng.random(labels.shape) < 0.02] = IGNORE
    tmp = Path(tempfile.mkdtemp())
    for name, noise in [("a", 3.0), ("b", 4.0)]:
        pred = (true + rng.normal(0, noise, true.shape)).astype(np.float32)
        logit = rng.normal(0, 1, (B, Q, N, 3)); logit[..., 2] += 2.0 * (labels == 2); logit[..., 1] += 1.0 * (labels == 1)
        np.savez(tmp / f"{name}_s0.npz", test_reg_pred=pred, test_reg_true=true, test_labels=labels,
                 test_logits=logit.astype(np.float16))
    va, vb = Variant(tmp / "a_s0.npz"), Variant(tmp / "b_s0.npz")
    one = np.ones((1, len(va.bl)))
    d = np.load(tmp / "a_s0.npz")
    # exact references
    ref_mae = reg_metrics(d["test_reg_pred"], d["test_reg_true"])["avg"]["mae"]
    lg = d["test_logits"]; pc = lg.argmax(-1); p2 = softmax_level2(lg)
    cm = cls_metrics(pc, d["test_labels"].astype(np.int64), prob_crit=p2)["avg"]
    print("MAE   blocks %.5f  exact %.5f" % (va.mae(one)[0], ref_mae))
    print("F1    blocks %.5f  exact %.5f" % (va.macro_f1(one)[0], cm["macro_f1"]))
    print("AP    binned %.5f  exact %.5f" % (va.ap2(one)[0], cm["pr_auc_level2"]))
    assert abs(va.mae(one)[0] - ref_mae) < 1e-6
    assert abs(va.macro_f1(one)[0] - cm["macro_f1"]) < 1e-6
    assert abs(va.ap2(one)[0] - cm["pr_auc_level2"]) < 5e-3
    r = compare(va, vb, "mae", 500, np.random.default_rng(1))
    print("a - b MAE:", {k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()})
    assert r["diff"] < 0 and r["ci_hi"] < 0, "a has less noise than b, the CI of a - b must be below zero"
    print("selftest ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results")
    ap.add_argument("--best", default="mtf_b1", help="variant name of the tuned MT-F, e.g. mtf_b2")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--out", default=None)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    else:
        run(a.dir, a.best, a.seed, a.reps, a.out)
