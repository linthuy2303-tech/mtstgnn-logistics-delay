"""Dataset statistics for Table 3 of the paper (numpy / pandas only).

Usage:
    python datastats.py metr-la data/METR-LA  [--json out.json]
    python datastats.py pems-bay data/PEMS-BAY

Reads the DCRNN-format .h5 file only (no adjacency needed). Labels use the default rule of the paper
(k = 3, thresholds 0.70 / 0.40, free-flow speed = 85th percentile of 00:00-05:00 speeds on the training split).
Missing readings are zeros, as in the DCRNN files.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from data import IGNORE, free_flow_speed, make_labels, split_ranges

FILES = {"metr-la": "metr-la.h5", "pems-bay": "pems-bay.h5"}


def shares(lab):
    lab = lab[lab != IGNORE]
    if lab.size == 0:
        return [float("nan")] * 3
    return (100.0 * np.bincount(lab, minlength=3)[:3] / lab.size).tolist()


def stats_from_df(df, P=12, Q=12):
    speed = df.values.astype(np.float32)
    idx = df.index
    tod = ((idx.hour * 60 + idx.minute).values / 1440.0).astype(np.float32)
    T, N = speed.shape
    _, tr_end = split_ranges(T, P, Q)
    va_end = int(T * 0.8)
    vf = free_flow_speed(speed, tod, tr_end)
    lab = make_labels(speed, vf)
    obs = speed > 0
    out = dict(
        n_sensors=int(N), n_steps=int(T),
        start=str(idx[0]), end=str(idx[-1]), days=float(T / 288.0),
        missing_pct=float(100.0 * (~obs).mean()),
        missing_pct_per_sensor_max=float(100.0 * (~obs).mean(0).max()),
        ignored_label_pct=float(100.0 * (lab == IGNORE).mean()),
        level_pct_all=shares(lab),
        level_pct_train=shares(lab[:tr_end]), level_pct_val=shares(lab[tr_end:va_end]), level_pct_test=shares(lab[va_end:]),
        speed_mean_train=float(speed[:tr_end][obs[:tr_end]].mean()), speed_std_train=float(speed[:tr_end][obs[:tr_end]].std()),
        v_free_median=float(np.median(vf)), v_free_min=float(vf.min()), v_free_max=float(vf.max()),
        split_steps=dict(train=int(tr_end), val=int(va_end - tr_end), test=int(T - va_end)),
        test_days=float((T - va_end) / 288.0),
    )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data", choices=sorted(FILES))
    ap.add_argument("data_dir")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    df = pd.read_hdf(Path(a.data_dir) / FILES[a.data])
    res = stats_from_df(df)
    txt = json.dumps(res, indent=1)
    print(txt)
    if a.json:
        Path(a.json).write_text(txt)


if __name__ == "__main__":
    main()
