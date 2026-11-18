"""Regression / classification metrics (numpy only)."""
import numpy as np

IGNORE = -100


def reg_metrics(pred, true, horizons=(3, 6, 12)):
    """pred, true: (B, Q, N) in raw units; true == 0 means missing. Horizon h = h*5 minutes."""
    def one(p, t):
        m = t > 0
        if m.sum() == 0:
            return dict(mae=float("nan"), rmse=float("nan"), mape=float("nan"))
        e = p[m] - t[m]
        return dict(mae=float(np.abs(e).mean()), rmse=float(np.sqrt((e ** 2).mean())),
                    mape=float((np.abs(e) / t[m]).mean() * 100))
    out = {"avg": one(pred, true)}
    for h in horizons:
        if h <= pred.shape[1]:
            out[f"h{h}"] = one(pred[:, h - 1], true[:, h - 1])
    return out


def _cls_one(p, t, n_cls):
    m = t != IGNORE
    p, t = p[m].astype(np.int64), t[m].astype(np.int64)
    cm = np.bincount(t * n_cls + p, minlength=n_cls * n_cls).reshape(n_cls, n_cls)  # rows = true
    tp = np.diag(cm).astype(np.float64)
    support, pred_n = cm.sum(1), cm.sum(0)
    recall = tp / np.maximum(support, 1)
    prec = tp / np.maximum(pred_n, 1)
    f1 = 2 * prec * recall / np.maximum(prec + recall, 1e-12)
    present = support > 0
    return dict(acc=float(tp.sum() / max(cm.sum(), 1)), macro_f1=float(f1[present].mean()),
                recall=recall.tolist(), precision=prec.tolist(), f1=f1.tolist(),
                support=support.tolist())


def average_precision(score, target, max_n=2_000_000, seed=0):
    """Non-interpolated average precision (area under PR curve) for a binary target."""
    score, target = score.ravel(), target.ravel().astype(bool)
    if len(score) > max_n:
        idx = np.random.default_rng(seed).choice(len(score), max_n, replace=False)
        score, target = score[idx], target[idx]
    if target.sum() == 0:
        return float("nan")
    y = target[np.argsort(-score, kind="stable")]
    prec = np.cumsum(y) / np.arange(1, len(y) + 1)
    return float((prec * y).sum() / y.sum())


def cls_metrics(pred, true, prob_crit=None, n_cls=3, horizons=(3, 6, 12)):
    """pred, true: (B, Q, N) int labels. prob_crit: (B, Q, N) probability of the top level."""
    out = {"avg": _cls_one(pred, true, n_cls)}
    for h in horizons:
        if h <= pred.shape[1]:
            out[f"h{h}"] = _cls_one(pred[:, h - 1], true[:, h - 1], n_cls)
    if prob_crit is not None:
        valid = true != IGNORE
        out["avg"]["pr_auc_level2"] = average_precision(prob_crit[valid], true[valid] == n_cls - 1)
    return out


def reg_by_level(pred, true, labels, n_cls=3):
    """Masked MAE of the speed forecast grouped by the *true* risk level of the target step.
    pred, true: (B, Q, N) raw units; labels: (B, Q, N) risk levels (IGNORE excluded)."""
    out = {}
    for c in range(n_cls):
        m = (labels == c) & (true > 0)
        out[f"level{c}"] = dict(mae=float(np.abs(pred[m] - true[m]).mean()) if m.any() else float("nan"), n=int(m.sum()))
    return out
