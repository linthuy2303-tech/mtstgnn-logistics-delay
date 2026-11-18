"""Early-warning evaluation (numpy only): calibration, event-based lead time, conformal intervals.

Input: the .npz written by `train.py --save_preds` (multi-task run).
  python eval_warning.py --preds results/multi_pcgrad.npz --out results/multi_pcgrad_warning.json

Windows in the npz are consecutive in time (stride 1), which lets us rebuild the label series
and detect disruption *onsets*. Time step = 5 min.
"""
import argparse
import json
import math

import numpy as np

IGNORE = -100
STEP_MIN = 5


# ----------------------------------------------------------------------------- calibration
def softmax(z, T=1.0):
    z = z.astype(np.float64) / T
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def _flat_valid(logits, labels):
    m = labels != IGNORE
    return logits[m].astype(np.float64), labels[m].astype(np.int64)


def nll(logits, labels, T=1.0):
    p = softmax(logits, T)
    return float(-np.log(np.clip(p[np.arange(len(labels)), labels], 1e-12, None)).mean())


def fit_temperature(logits, labels, lo=0.05, hi=20.0, iters=60):
    """Scalar temperature minimising NLL (convex in 1/T -> golden-section search on log T)."""
    z, y = _flat_valid(logits, labels)
    if len(y) > 2_000_000:  # subsample for speed
        idx = np.random.default_rng(0).choice(len(y), 2_000_000, replace=False)
        z, y = z[idx], y[idx]
    a, b = math.log(lo), math.log(hi)
    g = (math.sqrt(5) - 1) / 2
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = nll(z, y, math.exp(c)), nll(z, y, math.exp(d))
    for _ in range(iters):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a); fc = nll(z, y, math.exp(c))
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a); fd = nll(z, y, math.exp(d))
    return float(math.exp((a + b) / 2))


def ece_top_label(p, y, n_bins=15):
    conf, pred = p.max(-1), p.argmax(-1)
    acc = (pred == y).astype(np.float64)
    bins = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    e = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            e += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(e)


def ece_class(p_k, y_k, n_bins=15):
    """One-vs-rest ECE and reliability table for one class (e.g. level 2)."""
    bins = np.minimum((p_k * n_bins).astype(int), n_bins - 1)
    e, table = 0.0, []
    for b in range(n_bins):
        m = bins == b
        if m.any():
            e += m.mean() * abs(y_k[m].mean() - p_k[m].mean())
            table.append(dict(bin=b, n=int(m.sum()), conf=float(p_k[m].mean()), freq=float(y_k[m].mean())))
    return float(e), table


def brier_multiclass(p, y):
    onehot = np.eye(p.shape[-1])[y]
    return float(((p - onehot) ** 2).sum(-1).mean())


def calibration_report(logits_val, y_val, logits_test, y_test, n_bins=15):
    T = fit_temperature(logits_val, y_val)
    out = dict(temperature=T)
    zt, yt = _flat_valid(logits_test, y_test)
    for name, temp in (("raw", 1.0), ("scaled", T)):
        p = softmax(zt, temp)
        e2, tab = ece_class(p[:, -1], (yt == p.shape[-1] - 1).astype(float), n_bins)
        out[name] = dict(nll=nll(zt, yt, temp), brier=brier_multiclass(p, yt),
                         ece_top=ece_top_label(p, yt, n_bins), ece_level2=e2, reliability_level2=tab)
    return out


def calibration_by_horizon(logits_val, y_val, logits_test, y_test, T, horizons=(3, 6, 12)):
    """ECE / Brier per forecast step, raw vs scaled with the pooled temperature T."""
    out = {}
    for h in horizons:
        if h > logits_test.shape[1]:
            continue
        zt, yt = _flat_valid(logits_test[:, h - 1], y_test[:, h - 1])
        row = {}
        for name, temp in (("raw", 1.0), ("scaled", T)):
            p = softmax(zt, temp)
            row[name] = dict(ece_top=ece_top_label(p, yt), brier=brier_multiclass(p, yt),
                             ece_level2=ece_class(p[:, -1], (yt == p.shape[-1] - 1).astype(float))[0])
        out[f"h{h}"] = row
    return out


# ----------------------------------------------------------------------------- events / lead time
def label_series(labels):
    """labels: (W, Q, N) -> series y[time, N]; origin w predicts times w+1..w+Q."""
    W, Q, N = labels.shape
    y = np.full((W + Q, N), IGNORE, dtype=np.int64)
    for h in range(Q):
        y[h + 1:h + 1 + W] = labels[:, h]
    return y


def find_onsets(y, min_gap=6, crit=2):
    """Onset = first critical step after >= min_gap valid, non-critical steps."""
    is_c = (y == crit)
    valid = (y != IGNORE)
    c = np.concatenate([np.zeros((1, y.shape[1]), int), np.cumsum(is_c, 0)])      # c[u] = #crit in [0, u)
    v = np.concatenate([np.zeros((1, y.shape[1]), int), np.cumsum(valid, 0)])
    u = np.arange(y.shape[0])[:, None]
    lo = np.maximum(u - min_gap, 0)
    crit_before = np.take_along_axis(c, np.broadcast_to(u, y.shape), 0) - np.take_along_axis(c, np.broadcast_to(lo, y.shape), 0)
    valid_before = np.take_along_axis(v, np.broadcast_to(u, y.shape), 0) - np.take_along_axis(v, np.broadcast_to(lo, y.shape), 0)
    on = is_c & (crit_before == 0) & (valid_before == min_gap) & (u >= min_gap)
    return np.argwhere(on)  # (E, 2): [time, node]


def event_metrics(prob2, labels, theta, min_gap=6, leads_min=(15, 30, 45)):
    """prob2, labels: (W, Q, N). Returns event recall at several lead times, lead-time
    distribution and false-alert episodes per node-day."""
    W, Q, N = prob2.shape
    y = label_series(labels)
    ev = find_onsets(y, min_gap)
    res = dict(n_events=int(len(ev)), theta=float(theta))
    if len(ev):
        us, ns = ev[:, 0], ev[:, 1]
        ks = np.arange(1, Q + 1)
        w = us[:, None] - ks[None, :]                       # origin index for lead k
        ok = (w >= 0) & (w < W)
        P = np.where(ok, prob2[np.clip(w, 0, W - 1), ks[None, :] - 1, ns[:, None]], -1.0)
        hit = P >= theta
        for L in leads_min:
            k = L // STEP_MIN
            res[f"event_recall_{L}min"] = float(hit[:, k - 1:].any(1).mean()) if k <= Q else None
        kmax = (hit * ks[None, :]).max(1)
        det = kmax > 0
        res["event_recall_any"] = float(det.mean())
        res["median_lead_min"] = float(np.median(kmax[det]) * STEP_MIN) if det.any() else None
        res["lead_p25_min"] = float(np.percentile(kmax[det], 25) * STEP_MIN) if det.any() else None
        res["lead_p75_min"] = float(np.percentile(kmax[det], 75) * STEP_MIN) if det.any() else None
    # false-alert episodes: alert at origin (any horizon) but no critical step in the next Q steps
    alert = (prob2 >= theta).any(1)                         # (W, N)
    future_c = (labels == 2).any(1)
    fa = alert & ~future_c
    prev = np.vstack([np.zeros((1, N), bool), fa[:-1]])
    episodes = int((fa & ~prev).sum())
    node_days = (W / 288.0) * N
    res["false_alert_episodes"] = episodes
    res["false_alerts_per_node_day"] = float(episodes / node_days)
    return res


def step_metrics(prob2, labels, theta, horizons=(3, 6, 12)):
    """Per-step precision / recall / F1 of the level-2 alert at fixed horizons (not event-based)."""
    out = {}
    valid = labels != IGNORE
    for h in horizons:
        if h > prob2.shape[1]:
            continue
        p, y, v = prob2[:, h - 1] >= theta, labels[:, h - 1] == 2, valid[:, h - 1]
        p, y = p[v], y[v]
        tp = float((p & y).sum()); fp = float((p & ~y).sum()); fn = float((~p & y).sum())
        prec = tp / max(tp + fp, 1.0); rec = tp / max(tp + fn, 1.0)
        out[f"h{h}"] = dict(precision=prec, recall=rec, f1=2 * prec * rec / max(prec + rec, 1e-12))
    return out


def regression_score(pred_speed, v_free):
    """Pseudo-probability of level 2 from a speed forecast: 1 - forecast ratio, clipped to [0, 1]."""
    return np.clip(1.0 - pred_speed / v_free[None, None, :], 0.0, 1.0)


def select_threshold(prob2, labels, c_miss, c_fa, grid=None, min_gap=6, min_lead_min=15):
    """Pick theta minimising  c_miss * (#events not caught >= min_lead) + c_fa * (#false-alert episodes)."""
    grid = np.linspace(0.05, 0.95, 19) if grid is None else grid
    best = None
    for th in grid:
        m = event_metrics(prob2, labels, th, min_gap, leads_min=(min_lead_min,))
        r = m.get(f"event_recall_{min_lead_min}min")
        missed = (1 - (r if r is not None else 0.0)) * m["n_events"]
        cost = c_miss * missed + c_fa * m["false_alert_episodes"]
        if best is None or cost < best[0]:
            best = (cost, float(th))
    return best[1], best[0]


# ----------------------------------------------------------------------------- conformal
def conformal_report(pred_val, true_val, pred_test, true_test, alpha=0.1, horizons=(3, 6, 12)):
    """Split conformal: per-horizon quantile of |residual| on validation, coverage on test."""
    out = {}
    for h in horizons:
        if h > pred_val.shape[1]:
            continue
        mv, mt = true_val[:, h - 1] > 0, true_test[:, h - 1] > 0
        r = np.abs(pred_val[:, h - 1][mv] - true_val[:, h - 1][mv])
        n = len(r)
        q = float(np.quantile(r, min(1.0, math.ceil((n + 1) * (1 - alpha)) / n), method="higher"))
        cover = np.abs(pred_test[:, h - 1][mt] - true_test[:, h - 1][mt]) <= q
        out[f"h{h}"] = dict(q=q, coverage=float(cover.mean()), width=2 * q, nominal=1 - alpha)
    return out


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--alpha", type=float, default=0.1, help="conformal miscoverage level")
    ap.add_argument("--c_miss", type=float, default=10.0, help="cost of a missed disruption event")
    ap.add_argument("--cost_ratios", type=float, nargs="*", default=[5.0, 10.0, 20.0],
                    help="extra c_miss/c_fa ratios for the sensitivity sweep (c_fa fixed to --c_fa)")
    ap.add_argument("--c_fa", type=float, default=1.0, help="cost of a false-alert episode")
    ap.add_argument("--min_gap", type=int, default=6, help="quiet steps required before an onset")
    a = ap.parse_args()

    d = np.load(a.preds)
    res = {}
    if "val_logits" in d:
        lv, yv, lt, yt = d["val_logits"], d["val_labels"], d["test_logits"], d["test_labels"]
        res["calibration"] = cal = calibration_report(lv, yv, lt, yt)
        T = cal["temperature"]
        for name, temp in (("raw", 1.0), ("scaled", T)):
            pv, pt = softmax(lv, temp)[..., -1], softmax(lt, temp)[..., -1]
            th, cost = select_threshold(pv, yv, a.c_miss, a.c_fa, min_gap=a.min_gap)
            res[f"events_{name}"] = dict(val_cost=cost, **event_metrics(pt, yt, th, a.min_gap))
            res[f"steps_{name}"] = step_metrics(pt, yt, th)
        ps = softmax(lt, T)[..., -1]
        res["events_scaled_theta0.5"] = event_metrics(ps, yt, 0.5, a.min_gap)   # fixed-threshold reference
        res["steps_scaled_theta0.5"] = step_metrics(ps, yt, 0.5)
        res["calibration_by_horizon"] = calibration_by_horizon(lv, yv, lt, yt, T)
        res["cost_sweep"] = {}
        for name, temp in (("raw", 1.0), ("scaled", T)):
            pv, pt = softmax(lv, temp)[..., -1], softmax(lt, temp)[..., -1]
            for r in a.cost_ratios:
                th, cost = select_threshold(pv, yv, r * a.c_fa, a.c_fa, min_gap=a.min_gap)
                res["cost_sweep"][f"{name}_ratio{r:g}"] = dict(val_cost=cost, **event_metrics(pt, yt, th, a.min_gap))
    if "val_reg_pred" in d:
        res["conformal"] = conformal_report(d["val_reg_pred"], d["val_reg_true"],
                                            d["test_reg_pred"], d["test_reg_true"], a.alpha)
        if "v_free" in d:  # alert derived from the speed forecast alone (baseline without a classification head)
            vf = d["v_free"]
            sv, st = regression_score(d["val_reg_pred"], vf), regression_score(d["test_reg_pred"], vf)
            th, cost = select_threshold(sv, d["val_labels"], a.c_miss, a.c_fa, min_gap=a.min_gap)
            res["events_regression_derived"] = dict(val_cost=cost, **event_metrics(st, d["test_labels"], th, a.min_gap))
            res["steps_regression_derived"] = step_metrics(st, d["test_labels"], th)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps({k: v for k, v in res.items() if k.startswith("events_")}, indent=1))
    if "calibration" in res:
        cal = res["calibration"]
        print("temperature", cal["temperature"], "| ECE(top) raw/scaled",
              round(cal["raw"]["ece_top"], 4), round(cal["scaled"]["ece_top"], 4))


if __name__ == "__main__":
    main()
