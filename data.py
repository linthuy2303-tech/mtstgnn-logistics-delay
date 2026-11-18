"""Data loading, normalisation and risk-label construction (numpy only).

Supports:
  * METR-LA / PEMS-BAY in the DCRNN format (metr-la.h5 + adj_mx.pkl, pems-bay.h5 + adj_mx_bay.pkl)
  * a synthetic generator for smoke tests (no download needed)
"""
import pickle
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

IGNORE = -100  # label value for "missing / not usable"


# --------------------------------------------------------------------------- loaders
def make_synthetic(n_nodes=30, n_steps=6000, seed=0):
    """Fake 5-minute speed data with rush hours and diffusing congestion events."""
    rng = np.random.default_rng(seed)
    spd = 288  # steps per day
    t = np.arange(n_steps)
    tod = (t % spd) / spd
    dow = (t // spd) % 7
    v_free = rng.uniform(60, 70, n_nodes)

    pos = rng.uniform(0, 1, (n_nodes, 2))
    d = np.linalg.norm(pos[:, None] - pos[None], axis=-1)
    adj = np.exp(-((d / 0.25) ** 2))
    adj[adj < 0.1] = 0
    np.fill_diagonal(adj, 1.0)

    rush = 0.35 * np.exp(-(((tod - 0.33) / 0.05) ** 2)) + 0.40 * np.exp(-(((tod - 0.72) / 0.06) ** 2))
    rush = rush * (dow < 5)
    speed = v_free[None] * (1 - rush[:, None] * rng.uniform(0.5, 1.0, n_nodes)[None])

    drop = np.zeros((n_steps, n_nodes))
    for s in rng.integers(0, n_steps - 40, size=n_steps // 150):
        c = rng.integers(0, n_nodes)
        dur = int(rng.integers(6, 36))
        depth = rng.uniform(0.4, 0.85)
        w = adj[c] / adj[c].max()
        ramp = np.sin(np.linspace(0, np.pi, dur))
        drop[s:s + dur] += depth * ramp[:, None] * w[None]
    speed = speed * (1 - np.clip(drop, 0, 0.92)) + rng.normal(0, 1.0, speed.shape)
    speed = np.clip(speed, 3, None)
    speed[rng.random(speed.shape) < 0.01] = 0.0  # missing values are zeros, as in METR-LA
    return (speed.astype(np.float32), tod.astype(np.float32),
            dow.astype(np.float32), adj.astype(np.float32))


def load_dcrnn_h5(data_dir, h5_name, adj_name):
    import pandas as pd  # needs `tables` (pytables) for read_hdf
    df = pd.read_hdf(Path(data_dir) / h5_name)
    speed = df.values.astype(np.float32)
    idx = df.index
    tod = ((idx.hour * 60 + idx.minute).values / 1440.0).astype(np.float32)
    dow = idx.dayofweek.values.astype(np.float32)
    with open(Path(data_dir) / adj_name, "rb") as f:
        _, _, adj = pickle.load(f, encoding="latin1")
    return speed, tod, dow, np.asarray(adj, dtype=np.float32)


# --------------------------------------------------------------------------- labels
def free_flow_speed(speed, tod, train_end, night=(0.0, 5 / 24), q=85):
    """Per-node free-flow speed: q-th percentile of night-time speeds (train split only)."""
    sp = speed[:train_end].astype(np.float64).copy()
    sp[sp <= 0] = np.nan
    night_mask = (tod[:train_end] >= night[0]) & (tod[:train_end] < night[1])
    v = np.nanpercentile(sp[night_mask], q, axis=0)
    return np.where(np.isnan(v), np.nanmedian(v), v).astype(np.float32)


def make_labels(speed, v_free, k=3, t_warn=0.70, t_crit=0.40):
    """3-level risk label from the speed ratio v / v_free.

    Level 0 safe, 1 warning, 2 disruption. With k > 1 a level is assigned only if the
    ratio stayed below the threshold for k consecutive steps (persistence), so the label is
    not a plain per-step threshold of the regression target. k = 1 gives plain thresholds.
    Steps with missing data in the window get IGNORE.
    """
    ratio = np.where(speed > 0, speed / v_free[None], np.nan)
    T, N = ratio.shape
    padded = np.concatenate([np.full((k - 1, N), np.nan), ratio], axis=0)
    rmax = sliding_window_view(padded, k, axis=0).max(axis=-1)  # NaN propagates
    labels = np.zeros((T, N), dtype=np.int64)
    labels[rmax < t_warn] = 1
    labels[rmax < t_crit] = 2
    labels[np.isnan(rmax)] = IGNORE
    return labels


# --------------------------------------------------------------------------- splits
def split_ranges(T, P, Q, ratios=(0.7, 0.1, 0.2)):
    tr = int(T * ratios[0])
    va = int(T * (ratios[0] + ratios[1]))
    rng = lambda lo, hi: np.arange(lo, hi - P - Q + 1)
    return {"train": rng(0, tr), "val": rng(tr, va), "test": rng(va, T)}, tr


def prepare(name, data_dir=".", P=12, Q=12, persist_k=3, seed=0, t_warn=0.70, t_crit=0.40):
    if name == "synthetic":
        speed, tod, dow, adj = make_synthetic(seed=seed)
    elif name == "metr-la":
        speed, tod, dow, adj = load_dcrnn_h5(data_dir, "metr-la.h5", "adj_mx.pkl")
    elif name == "pems-bay":
        speed, tod, dow, adj = load_dcrnn_h5(data_dir, "pems-bay.h5", "adj_mx_bay.pkl")
    else:
        raise ValueError(name)

    T, N = speed.shape
    starts, train_end = split_ranges(T, P, Q)
    obs = speed > 0
    vals = speed[:train_end][obs[:train_end]]
    mean, std = float(vals.mean()), float(vals.std())

    speed_n = np.where(obs, (speed - mean) / std, 0.0)
    feats = np.stack([
        speed_n,
        np.repeat(tod[:, None], N, axis=1),
        np.repeat((dow / 6.0)[:, None], N, axis=1),
        obs.astype(np.float32),
    ], axis=-1).astype(np.float32)  # (T, N, C=4)

    v_free = free_flow_speed(speed, tod, train_end)
    labels = make_labels(speed, v_free, k=persist_k, t_warn=t_warn, t_crit=t_crit)
    lt = labels[:train_end]
    counts = np.bincount(lt[lt != IGNORE], minlength=3).astype(np.float64)

    return dict(feats=feats, speed=speed.astype(np.float32), labels=labels, adj=adj,
                starts=starts, mean=mean, std=std, v_free=v_free,
                class_counts=counts, n_nodes=N, n_feats=feats.shape[-1])
