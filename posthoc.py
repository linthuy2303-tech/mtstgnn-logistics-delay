"""Per-run post-hoc metrics from a saved predictions file (numpy only).

Usage:
    python posthoc.py --npz results/reg_s0.npz --out results/reg_s0_post.json

Needs the arrays written by `train.py --save_preds`:
    test_reg_pred, test_reg_true (B, Q, N, mph; true == 0 means missing), test_labels (B, Q, N, int8; -100 = ignore),
    v_free (N,).  Runs without a regression head (mode cls) have no speed arrays and are skipped.

What it adds to the numbers train.py already prints
  * speed MAE averaged over the 12 steps and at 15 / 30 / 60 min   (comparable with BasicTS+ "average" numbers)
  * delay MAE in minutes per mile:  |60/v_hat - 60/v|  (v_free cancels), speed clipped at 1 mph
  * speed and delay MAE by TRUE risk level (0 / 1 / 2)
  * R->thr baseline: the speed forecast is turned into a risk level with the same rule as the labels
    (ratio = forecast / v_free, worst ratio over the last k forecast steps, thresholds theta1 / theta2); score for level 2 = 1 - ratio.
    NOTE: the persistence window is truncated at the first forecast step (the npz does not hold the observed past),
    so for horizons q < k fewer than k steps are used.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from metrics import cls_metrics, reg_metrics, IGNORE


def delay(v):
    return 60.0 / np.maximum(v, 1.0)


def delay_metrics(pred, true, horizons=(3, 6, 12)):
    m = true > 0
    out = {"mae_avg": float(np.abs(delay(pred[m]) - delay(true[m])).mean())}
    for h in horizons:
        if h <= pred.shape[1]:
            mh = true[:, h - 1] > 0
            out[f"mae_h{h}"] = float(np.abs(delay(pred[:, h - 1][mh]) - delay(true[:, h - 1][mh])).mean())
    return out


def by_level(pred, true, labels, n_cls=3):
    sp, dl, n = [], [], []
    for c in range(n_cls):
        m = (true > 0) & (labels == c)
        n.append(int(m.sum()))
        if m.sum() == 0:
            sp.append(float("nan")); dl.append(float("nan")); continue
        sp.append(float(np.abs(pred[m] - true[m]).mean()))
        dl.append(float(np.abs(delay(pred[m]) - delay(true[m])).mean()))
    return dict(speed_mae=sp, delay_mae=dl, n=n)


def forecast_levels(pred, v_free, k=3, theta1=0.70, theta2=0.40):
    """Apply the labelling rule to the forecast trajectory. Returns (level (B,Q,N) int, score (B,Q,N))."""
    ratio = pred / v_free[None, None, :]
    q_len = ratio.shape[1]
    m = np.empty_like(ratio)
    for q in range(q_len):
        m[:, q] = ratio[:, max(0, q - k + 1): q + 1].max(axis=1)
    level = np.where(m < theta2, 2, np.where(m < theta1, 1, 0)).astype(np.int8)
    score = np.clip(1.0 - m, 0.0, 1.0)
    return level, score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--theta1", type=float, default=0.70)
    ap.add_argument("--theta2", type=float, default=0.40)
    a = ap.parse_args()

    d = np.load(a.npz)
    if "test_reg_pred" not in d:
        print("no regression predictions in", a.npz, "- nothing to do")
        return
    pred, true, labels, vf = d["test_reg_pred"], d["test_reg_true"], d["test_labels"].astype(np.int64), d["v_free"]
    res = dict(
        speed={("mae_avg" if k == "avg" else f"mae_{k}"): v["mae"] for k, v in reg_metrics(pred, true).items()},
        delay=delay_metrics(pred, true),
        by_level=by_level(pred, true, labels),
    )
    level, score = forecast_levels(pred, vf, a.k, a.theta1, a.theta2)
    res["r_thr"] = cls_metrics(level, labels, prob_crit=score)
    res["params"] = dict(k=a.k, theta1=a.theta1, theta2=a.theta2)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2))
    print("speed", {k: round(v, 3) for k, v in res["speed"].items()})
    print("delay", {k: round(v, 3) for k, v in res["delay"].items()})
    print("R->thr macro-F1 %.3f  recall L2 %.3f  AP L2 %.3f" % (
        res["r_thr"]["avg"]["macro_f1"], res["r_thr"]["avg"]["recall"][2],
        res["r_thr"]["avg"].get("pr_auc_level2", float("nan"))))


if __name__ == "__main__":
    main()
