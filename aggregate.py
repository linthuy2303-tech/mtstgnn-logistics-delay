"""Aggregate many seeds into the tables of the paper (mean +- sd) and run the pre-registered checks.

Expected files in results/ (written by run_plan.sh):
    <variant>_s<seed>.json         from train.py          (variants: reg, cls, mtf_b0.25 ... mtf_b4, mtu, mtp, ...)
    <variant>_s<seed>_post.json    from posthoc.py        (only runs with a regression head)
Usage:
    python aggregate.py results/ > results/summary.md                      # markdown, for reading
    python aggregate.py results/ --json results/summary.json > results/summary.md   # + the same numbers as JSON
Notes
  * "MT-F (best beta)" = the mtf_b* variant with the lowest mean best validation loss (unweighted sum of the two losses),
    which is the same criterion used for early stopping. It is chosen on validation data only.
  * R + C (separate) pairs reg_s<i> with cls_s<i>; parameters and time are summed.
  * Gradient cosines are subsampled to every 20th step for every variant (PCGrad logs every step).
  * Wilcoxon signed-rank tests are paired by seed (exact distribution); with fewer than 6 seeds p < 0.05 is unreachable.
  * Decision rules of H1-H4 are fixed by the constants below (they were written before any result existed).
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    from scipy.stats import wilcoxon
except Exception:  # scipy missing
    wilcoxon = None

MAC_MT, MAC_RC = 405.234378, 808.98498  # analytic MACs per sample (millions), METR-LA, d=32, L=3 (paper, Table 9)
TOL_SD = 1.0            # H1 "matches" and H3 "beaten by more than" are judged in units of one seed standard deviation
H2_MIN_SHARE = 0.10     # H2: share of logged steps with a negative gradient cosine, for MT-F (beta = 1)
H4_RANGE = (0.40, 0.60)  # H4: "about half" = ratio MT / (R + C) inside this range, for parameters and for MACs


# --------------------------------------------------------------------------- small helpers
def fin(v):
    return [float(x) for x in v if x is not None and np.isfinite(x)]


def mean_sd(v):
    v = fin(v)
    if not v:
        return None
    return dict(mean=float(np.mean(v)), sd=float(np.std(v, ddof=1)) if len(v) > 1 else 0.0, n=len(v))


def ms(v, nd=3, pct=False):
    v = fin(v)
    if pct:
        v = [x * 100 for x in v]
    s = mean_sd(v)
    if s is None:
        return "n/a"
    if s["n"] == 1:  # a single seed has no spread to report
        return f"{s['mean']:.{nd}f}"
    return f"{s['mean']:.{nd}f} ± {s['sd']:.{nd}f}"


def load(d):
    runs = defaultdict(dict)
    for p in sorted(Path(d).glob("*_s*.json")):
        m = re.match(r"^(.+)_s(\d+)$", p.stem)
        if not m or p.stem.endswith("_post"):
            continue
        var, seed = m.group(1), int(m.group(2))
        r = json.loads(p.read_text())
        post = Path(d) / f"{p.stem}_post.json"
        r["post"] = json.loads(post.read_text()) if post.exists() else None
        runs[var][seed] = r
    return runs


def get(r, path, default=None):
    cur = r
    for k in path:
        try:
            cur = cur[k]
        except (KeyError, IndexError, TypeError):
            return default
    return cur


def best_val(r):
    return min(h["val"] for h in r["history"]) if r.get("history") else float("nan")


def best_epoch(r):
    return int(np.argmin([h["val"] for h in r["history"]])) + 1


def metrics_of(r, kind="own"):
    """Flat dict of the headline numbers of one run. kind='thr' reads the R->thr block of a regression run."""
    t = r["test"]
    out = {}
    if kind == "own":
        out["mae30"] = get(t, ["reg", "h6", "mae"])
        out["mae_avg"] = get(t, ["reg", "avg", "mae"])
        out["delay"] = get(r, ["post", "delay", "mae_avg"])
        out["f1"] = get(t, ["cls", "avg", "macro_f1"])
        out["rec2"] = get(t, ["cls", "avg", "recall", 2])
        out["ap2"] = get(t, ["cls", "avg", "pr_auc_level2"])
    else:
        out["f1"] = get(r, ["post", "r_thr", "avg", "macro_f1"])
        out["rec2"] = get(r, ["post", "r_thr", "avg", "recall", 2])
        out["ap2"] = get(r, ["post", "r_thr", "avg", "pr_auc_level2"])
    return out


def cos_stats(r):
    log = [(e, b, c) for e, b, c in r.get("grad_cos", []) if b % 20 == 0]
    if not log:
        return None
    c = np.array([x[2] for x in log]); e = np.array([x[0] for x in log])
    early, late = c[e <= 5], c[e > 5]
    return dict(mean=float(c.mean()), neg=float((c < 0).mean()),
                neg_early=float((early < 0).mean()) if len(early) else float("nan"),
                neg_late=float((late < 0).mean()) if len(late) else float("nan"))


def seeds_common(*vs):
    s = set.intersection(*[set(v) for v in vs]) if vs else set()
    return sorted(s)


def clean(o):
    """JSON-safe copy: numpy -> python, NaN/inf -> None."""
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, np.integer):
        return int(o)
    return o


# --------------------------------------------------------------------------- the whole analysis
def analyse(runs, eff_seeds=None):
    T, NUM, X, CHK = {}, {}, {}, []
    order = []

    def table(key, title, header):
        T[key] = dict(title=title, header=header, rows=[])
        order.append(key)

    def add(key, rid, label, cells):
        T[key]["rows"].append(dict(id=rid, label=label, cells=cells))

    X["seeds"] = {k: len(v) for k, v in sorted(runs.items())}

    # ---- choose the best beta for MT-F by mean best validation loss
    mtf = {k: v for k, v in runs.items() if k.startswith("mtf_b")}
    best_mtf, scores = None, {}
    if mtf:
        # beta is compared on the seeds that every beta has (so unequal numbers of seeds cannot decide the choice)
        common = seeds_common(*mtf.values())
        scores = {k: float(np.mean(fin([best_val(r) for s, r in v.items() if not common or s in common])))
                  for k, v in mtf.items()}
        best_mtf = min(scores, key=scores.get)
        X["val_by_beta"] = {k: round(x, 4) for k, x in sorted(scores.items(), key=lambda kv: float(kv[0][5:]))}
        X["beta_selection_seeds"] = common
        X["best_mtf"] = best_mtf
        X["best_beta"] = float(best_mtf[5:])

    # ---- per-variant numeric summaries (used by the checks and by the paper text)
    def seed_vals(var, key, kind="own"):
        out = {}
        for s in sorted(runs.get(var, {})):
            v = metrics_of(runs[var][s], kind).get(key)
            if v is not None and np.isfinite(v):
                out[s] = float(v)
        return out

    mt_ids = {"MT-F (β = 1)": "mtf_b1", "MT-F (best β)": best_mtf, "MT-U": "mtu", "MT-P": "mtp"}
    for name, var, kind in [("R", "reg", "own"), ("R→thr", "reg", "thr"), ("C", "cls", "own")] + \
            [(n, v, "own") for n, v in mt_ids.items() if v]:
        if var not in runs:
            continue
        keys = ("mae30", "mae_avg", "delay", "f1", "rec2", "ap2") if kind == "own" else ("f1", "rec2", "ap2")
        NUM[name] = {k: dict(**(mean_sd(list(seed_vals(var, k, kind).values())) or {}),
                             values=list(seed_vals(var, k, kind).values()),
                             seeds=list(seed_vals(var, k, kind).keys())) for k in keys}

    # ---- Table 5
    table("t5", "Table 5 – test performance",
          ["Model", "MAE30", "Avg MAE", "Delay MAE", "Macro-F1", "Recall L2", "AP L2"])
    rows5 = [("R", "R", "reg", "own"), ("R→thr", "R→thr", "reg", "thr"), ("C", "C", "cls", "own")]
    if "mtf_b1" in runs:
        rows5.append(("MT-F (β = 1)", "MT-F (β = 1)", "mtf_b1", "own"))
    if best_mtf:
        rows5.append(("MT-F (best β)", f"MT-F (best β: {best_mtf})", best_mtf, "own"))
    rows5 += [("MT-U", "MT-U", "mtu", "own"), ("MT-P", "MT-P", "mtp", "own")]
    for rid, label, var, kind in rows5:
        if var not in runs:
            continue
        ms_ = [metrics_of(runs[var][s], kind) for s in sorted(runs[var])]
        col = lambda k, nd=3: ms([m.get(k) for m in ms_], nd)
        if kind == "thr":
            add("t5", rid, label, ["= R", "= R", "= R", col("f1"), col("rec2"), col("ap2")])
        else:
            add("t5", rid, label, [col("mae30", 2), col("mae_avg", 2), col("delay", 3), col("f1"), col("rec2"), col("ap2")])
    if "reg" in runs and "cls" in runs:
        pair = seeds_common(runs["reg"], runs["cls"])
        a = [metrics_of(runs["reg"][s]) for s in pair]; b = [metrics_of(runs["cls"][s]) for s in pair]
        add("t5", "R + C", "R + C (separate)",
            [ms([x["mae30"] for x in a], 2), ms([x["mae_avg"] for x in a], 2), ms([x["delay"] for x in a]),
             ms([x["f1"] for x in b]), ms([x["rec2"] for x in b]), ms([x["ap2"] for x in b])])

    # ---- Table 6: by horizon
    table("t6", "Table 6 – speed MAE and macro-F1 by horizon",
          ["Model", "MAE 15", "MAE 30", "MAE 60", "F1 15", "F1 30", "F1 60"])
    for rid, var in [("R", "reg"), ("C", "cls"), ("MT-F (best β)", best_mtf), ("MT-U", "mtu"), ("MT-P", "mtp")]:
        if not var or var not in runs:
            continue
        cells = []
        for h in ("h3", "h6", "h12"):
            cells.append(ms([get(runs[var][s], ["test", "reg", h, "mae"]) for s in sorted(runs[var])], 2))
        for h in ("h3", "h6", "h12"):
            cells.append(ms([get(runs[var][s], ["test", "cls", h, "macro_f1"]) for s in sorted(runs[var])]))
        add("t6", rid, rid, cells)

    # ---- Table 7: per class, 30 min
    table("t7", "Table 7 – per-class precision / recall / F1 at 30 min", ["Model", "Level", "Precision", "Recall", "F1"])
    for rid, var in [("C", "cls"), ("MT-P", "mtp")]:
        if var not in runs:
            continue
        for c in range(3):
            cell = [ms([get(runs[var][s], ["test", "cls", "h6", k, c]) for s in sorted(runs[var])])
                    for k in ("precision", "recall", "f1")]
            add("t7", f"{rid}|{c}", f"{rid} | level {c}", cell)

    # ---- Table 8: gradient diagnostics
    table("t8", "Table 8 – gradient cosine on the shared encoder (every 20th step, before projection)",
          ["Strategy", "Mean cos φ", "cos φ < 0 (%), epochs 1–5", "cos φ < 0 (%), later", "cos φ < 0 (%), all"])
    for rid, var in [("MT-F (β = 1)", "mtf_b1"), ("MT-F (best β)", best_mtf), ("MT-U", "mtu"), ("MT-P", "mtp")]:
        if not var or var not in runs:
            continue
        cs = [c for c in (cos_stats(runs[var][s]) for s in sorted(runs[var])) if c]
        if not cs:
            add("t8", rid, rid, ["(no grad_cos logged)", "", "", ""]); continue
        add("t8", rid, rid, [ms([c["mean"] for c in cs]), ms([c["neg_early"] for c in cs], 1, True),
                             ms([c["neg_late"] for c in cs], 1, True), ms([c["neg"] for c in cs], 1, True)])

    # ---- Table 9: efficiency
    table("t9", "Table 9 – cost (measured; training time per epoch in the mode the runs used, latency always float32"
          + (f"; seeds {', '.join(str(x) for x in sorted(eff_seeds))} only, the runs made on one machine" if eff_seeds else "") + ")",
          ["Model", "Params", "s / epoch", "Latency ms (batch 1)", "Peak GPU MB"])
    # --eff-seeds restricts the cost table to the seeds that were run on one machine (timings of other machines are not comparable)
    es = lambda var: [x for x in sorted(runs[var]) if eff_seeds is None or x in eff_seeds] or sorted(runs[var])
    eff = lambda var, k: [get(runs[var][x], ["efficiency", k]) for x in es(var)]
    for rid, var in [("R", "reg"), ("C", "cls")]:
        if var in runs:
            add("t9", rid, rid, [f"{int(np.mean(eff(var, 'params'))):,}", ms(eff(var, "train_s_per_epoch"), 1),
                                 ms(eff(var, "latency_ms_batch1"), 2), ms(eff(var, "peak_gpu_mb"), 0)])
    ts_rc = None
    if "reg" in runs and "cls" in runs:
        pair = seeds_common(runs["reg"], runs["cls"])
        pair = [x for x in pair if eff_seeds is None or x in eff_seeds] or pair
        ts_rc = [runs["reg"][s]["efficiency"]["train_s_per_epoch"] + runs["cls"][s]["efficiency"]["train_s_per_epoch"] for s in pair]
        lt = [runs["reg"][s]["efficiency"]["latency_ms_batch1"] + runs["cls"][s]["efficiency"]["latency_ms_batch1"] for s in pair]
        pr = runs["reg"][pair[0]]["efficiency"]["params"] + runs["cls"][pair[0]]["efficiency"]["params"]
        add("t9", "R + C", "R + C (sum)", [f"{pr:,}", ms(ts_rc, 1), ms(lt, 2), "–"])
        X["params_rc"] = pr
    for rid, var in [("MT-F (β = 1)", "mtf_b1"), ("MT-U", "mtu"), ("MT-P", "mtp")]:
        if var in runs:
            add("t9", rid, rid, [f"{int(np.mean(eff(var, 'params'))):,}", ms(eff(var, "train_s_per_epoch"), 1),
                                 ms(eff(var, "latency_ms_batch1"), 2), ms(eff(var, "peak_gpu_mb"), 0)])
    if "mtp" in runs:
        X["params_mt"] = int(runs["mtp"][sorted(runs["mtp"])[0]]["efficiency"]["params"])
    if "params_rc" in X and "params_mt" in X:
        X["param_ratio"] = X["params_mt"] / X["params_rc"]
        X["mac_ratio"] = MAC_MT / MAC_RC
    if ts_rc:
        for var in ("mtf_b1", "mtu", "mtp"):
            if var in runs:
                X.setdefault("time_ratio_to_RC", {})[var] = float(np.mean(eff(var, "train_s_per_epoch")) / np.mean(ts_rc))
    X["eff_seeds"] = sorted(eff_seeds) if eff_seeds else None
    X["amp"] = bool(any(get(r, ["efficiency", "amp"], False) for v in runs.values() for r in v.values()))
    X["amp_skipped_steps"] = int(sum(get(r, ["efficiency", "amp_skipped_steps"], 0) or 0
                                     for v in runs.values() for r in v.values()))
    X["runs_total"] = int(sum(len(v) for v in runs.values()))

    # ---- Table 10: by true level
    table("t10", "Table 10 – speed MAE (mph) and delay MAE (min/mile) by true risk level",
          ["Model", "Speed L0", "Speed L1", "Speed L2", "Delay L0", "Delay L1", "Delay L2"])
    for rid, var in [("R", "reg"), ("MT-F", best_mtf), ("MT-U", "mtu"), ("MT-P", "mtp")]:
        if not var or var not in runs:
            continue
        cells = []
        for key in ("speed_mae", "delay_mae"):
            for c in range(3):
                cells.append(ms([get(runs[var][s], ["post", "by_level", key, c]) for s in sorted(runs[var])], 3))
        add("t10", rid, rid, cells)

    # ---- Table 11: sensitivity and ablation (all rows use the seeds that every sensitivity variant has, e.g. seed 0 only)
    sens_vars = [k for k in runs if re.match(r"^(cls|mtp)_(k1|k6|t60_30|t80_50|noadj|nocw)$", k)]
    t11_seeds = seeds_common(*[runs[k] for k in sens_vars]) if sens_vars else None
    X["t11_seeds"] = t11_seeds
    sv = lambda var: [s for s in sorted(runs.get(var, {})) if t11_seeds is None or s in t11_seeds]
    table("t11", "Table 11 – sensitivity to the labelling rule and ablation",
          ["(k, θ1, θ2) / variant", "L2 share (%)", "F1 (C)", "F1 (MT-P)", "AP L2 (MT-P)"])
    cfgs = [("(1, 0.70, 0.40)", "k1"), ("(3, 0.70, 0.40), default", ""), ("(6, 0.70, 0.40)", "k6"),
            ("(3, 0.60, 0.30)", "t60_30"), ("(3, 0.80, 0.50)", "t80_50")]
    for label, tag in cfgs:
        c = "cls" + (f"_{tag}" if tag else ""); m = "mtp" + (f"_{tag}" if tag else "")
        if c not in runs and m not in runs:
            continue
        src = runs.get(m) or runs.get(c)
        cc = np.array([r.get("class_counts", [0, 0, 0]) for r in src.values()], dtype=float).mean(0)
        share = f"{100 * cc[2] / cc.sum():.3f}" if cc.sum() > 0 else "n/a"
        f1c = ms([get(runs[c][s], ["test", "cls", "avg", "macro_f1"]) for s in sv(c)])
        f1m = ms([get(runs[m][s], ["test", "cls", "avg", "macro_f1"]) for s in sv(m)])
        apm = ms([get(runs[m][s], ["test", "cls", "avg", "pr_auc_level2"]) for s in sv(m)])
        add("t11", tag or "default", label, [share, f1c, f1m, apm])
    for label, tag in [("No adaptive adjacency", "mtp_noadj"), ("No class weights", "mtp_nocw")]:
        if tag in runs:
            f1m = ms([get(runs[tag][s], ["test", "cls", "avg", "macro_f1"]) for s in sv(tag)])
            apm = ms([get(runs[tag][s], ["test", "cls", "avg", "pr_auc_level2"]) for s in sv(tag)])
            add("t11", tag, label, ["–", "–", f1m, apm])

    # ---- Table 12: dynamics
    table("t12", "Table 12 – training dynamics", ["Model", "Best epoch", "Final train loss", "Best val loss", "Epochs run"])
    for rid, var in [("R", "reg"), ("C", "cls"), ("MT-F (best β)", best_mtf), ("MT-U", "mtu"), ("MT-P", "mtp")]:
        if not var or var not in runs:
            continue
        ss = sorted(runs[var])
        add("t12", rid, rid, [ms([best_epoch(runs[var][s]) for s in ss], 1),
                              ms([runs[var][s]["history"][-1]["train"] for s in ss], 3),
                              ms([best_val(runs[var][s]) for s in ss], 3),
                              ms([len(runs[var][s]["history"]) for s in ss], 1)])

    # ---- class shares (training split of the default labelling)
    for var in ("cls", "mtp", "reg"):
        if var in runs:
            cc = np.array([r.get("class_counts", [0, 0, 0]) for r in runs[var].values()], dtype=float).mean(0)
            if cc.sum() > 0:
                X["train_level_shares_pct"] = [float(100 * x / cc.sum()) for x in cc]
            break

    # ---- Pre-registered checks (rules fixed by the constants at the top of this file)
    def paired(a, b, key, kind_a="own", kind_b="own"):
        """a - b on one metric, paired by seed."""
        if a not in runs or b not in runs:
            return None
        va, vb = seed_vals(a, key, kind_a), seed_vals(b, key, kind_b)
        common = sorted(set(va) & set(vb))
        if not common:
            return None
        x = np.array([va[s] for s in common]); y = np.array([vb[s] for s in common])
        diff = x - y
        p = float("nan")
        if wilcoxon is not None and len(x) >= 2 and np.any(diff != 0):
            try:
                p = float(wilcoxon(x, y, method="exact").pvalue)
            except Exception:
                p = float(wilcoxon(x, y).pvalue)
        return dict(a=a, b=b, key=key, n=len(x), mean_a=float(x.mean()), mean_b=float(y.mean()),
                    diff=float(diff.mean()), sd_a=float(x.std(ddof=1)) if len(x) > 1 else 0.0,
                    sd_b=float(y.std(ddof=1)) if len(y) > 1 else 0.0, p=p,
                    a_lower=int((diff < 0).sum()), a_higher=int((diff > 0).sum()))

    mt_for_checks = [("MT-F (best β)", best_mtf), ("MT-U", "mtu"), ("MT-P", "mtp")]
    # H1: best multi-task variant (lowest mean validation loss among the three) matches R on MAE and C on AP L2
    cand = {n: v for n, v in mt_for_checks if v and v in runs}
    if cand and "reg" in runs and "cls" in runs:
        vloss = {n: float(np.mean(fin([best_val(r) for r in runs[v].values()]))) for n, v in cand.items()}
        best_name = min(vloss, key=vloss.get)
        X["best_mt_by_val"] = best_name
        for n, v in cand.items():
            a = paired(v, "reg", "mae_avg"); b = paired(v, "cls", "ap2")
            if a and b:
                ok_mae = a["diff"] <= TOL_SD * a["sd_b"]          # not worse than R by more than one seed sd of R
                ok_ap = b["diff"] >= -TOL_SD * b["sd_b"]          # not worse than C by more than one seed sd of C
                CHK.append(dict(id="H1", variant=n, selected=(n == best_name), mae=a, ap2=b,
                                ok_mae=bool(ok_mae), ok_ap2=bool(ok_ap), supported=bool(ok_mae and ok_ap)))
    # H2
    if "mtf_b1" in runs:
        cs = [c for c in (cos_stats(runs["mtf_b1"][s]) for s in sorted(runs["mtf_b1"])) if c]
        if cs:
            share = float(np.mean([c["neg"] for c in cs]))
            CHK.append(dict(id="H2", share=share, criterion=H2_MIN_SHARE, supported=bool(share >= H2_MIN_SHARE)))
    # H3: tuned MT-F is not beaten by MT-U / MT-P by more than one seed sd of tuned MT-F
    ref = best_mtf or ("mtf_b1" if "mtf_b1" in runs else None)
    if ref:
        for n, v in [("MT-U", "mtu"), ("MT-P", "mtp")]:
            a = paired(v, ref, "mae_avg"); b = paired(v, ref, "ap2")
            if a and b:
                beaten_mae = (-a["diff"]) > TOL_SD * a["sd_b"]    # other has lower error by more than one sd of the reference
                beaten_ap = b["diff"] > TOL_SD * b["sd_b"]        # other has higher AP by more than one sd of the reference
                CHK.append(dict(id="H3", variant=n, reference=ref, mae=a, ap2=b,
                                beaten_mae=bool(beaten_mae), beaten_ap2=bool(beaten_ap),
                                supported=bool(not beaten_mae and not beaten_ap)))
    # H4
    if "param_ratio" in X:
        ok = H4_RANGE[0] <= X["param_ratio"] <= H4_RANGE[1] and H4_RANGE[0] <= X["mac_ratio"] <= H4_RANGE[1]
        CHK.append(dict(id="H4", param_ratio=X["param_ratio"], mac_ratio=X["mac_ratio"], range=list(H4_RANGE),
                        supported=bool(ok)))
    # extra, descriptive (not part of H1-H4): MT variants vs R / C and best MT-F, with Wilcoxon p
    extra = []
    for n, v in mt_for_checks:
        if not v:
            continue
        for ref_, key, label in [("reg", "mae_avg", "vs R"), ("cls", "ap2", "vs C"), ("cls", "f1", "vs C"),
                                 (best_mtf, "mae_avg", "vs MT-F (best β)"), (best_mtf, "ap2", "vs MT-F (best β)"),
                                 (best_mtf, "f1", "vs MT-F (best β)")]:
            if ref_ and ref_ != v:
                r = paired(v, ref_, key)
                if r:
                    r["label"] = f"{n} {label}"
                    extra.append(r)
    return dict(order=order, tables=T, num=NUM, notes=X, checks=CHK, pairs=extra)


# --------------------------------------------------------------------------- output
def to_markdown(res, d):
    X = res["notes"]
    out = ["# Summary (mean ± sd over seeds)\n", f"Directory: {d}\n",
           f"Seeds per variant: {X['seeds']}\n"]
    if "val_by_beta" in X:
        out.append(f"MT-F validation loss by beta (seeds {X.get('beta_selection_seeds')}): {X['val_by_beta']} -> best: {X['best_mtf']}\n")
    for key in res["order"]:
        t = res["tables"][key]
        out.append(f"\n## {t['title']}\n")
        out.append("| " + " | ".join(t["header"]) + " |\n|" + "---|" * len(t["header"]))
        for r in t["rows"]:
            out.append("| " + " | ".join([r["label"]] + r["cells"]) + " |")
        if key == "t11" and X.get("t11_seeds"):
            out.append(f"\nSeeds used in every row of Table 11: {X['t11_seeds']}")
        if key == "t9" and "param_ratio" in X:
            out.append(f"\nParameter ratio MT / (R + C) = {X['param_ratio']:.3f}; analytic MAC ratio = {X['mac_ratio']:.3f}")
            for v, x in X.get("time_ratio_to_RC", {}).items():
                out.append(f"Training time per epoch {v} / (R + C) = {x:.3f}")
    if "train_level_shares_pct" in X:
        out.append("\nTraining-split level shares 0/1/2 (%%): %s" % " / ".join(f"{x:.2f}" for x in X["train_level_shares_pct"]))
    out.append(f"\nfloat16 encoder used: {X['amp']}; skipped (overflow) steps in all runs: {X['amp_skipped_steps']}; runs: {X['runs_total']}")
    out.append("\n## Hypothesis checks (paired by seed; rules in the constants of aggregate.py)\n")
    out.append(f"Best multi-task variant by mean validation loss: {X.get('best_mt_by_val', 'n/a')}\n")
    pf = lambda r: (f"{r['key']}: {r['a']} {r['mean_a']:.4f} vs {r['b']} {r['mean_b']:.4f}, diff {r['diff']:+.4f} "
                    f"(seed sd {r['sd_a']:.4f} / {r['sd_b']:.4f}), exact Wilcoxon p = {r['p']:.3f}, n = {r['n']}")
    for c in res["checks"]:
        if c["id"] == "H1":
            out.append(f"- H1 [{c['variant']}{' (selected)' if c['selected'] else ''}]: {pf(c['mae'])}; {pf(c['ap2'])} "
                       f"-> MAE {'ok' if c['ok_mae'] else 'worse'}, AP {'ok' if c['ok_ap2'] else 'worse'} "
                       f"=> {'supported' if c['supported'] else 'not supported'}")
        elif c["id"] == "H2":
            out.append(f"- H2: share of logged steps with cos φ < 0 for MT-F (β = 1): {c['share'] * 100:.1f}% "
                       f"(criterion ≥ {c['criterion'] * 100:.0f}%) => {'supported' if c['supported'] else 'not supported'}")
        elif c["id"] == "H3":
            out.append(f"- H3 [{c['variant']} vs {c['reference']}]: {pf(c['mae'])}; {pf(c['ap2'])} "
                       f"-> beaten on MAE: {c['beaten_mae']}, on AP: {c['beaten_ap2']} "
                       f"=> {'supported' if c['supported'] else 'not supported'}")
        elif c["id"] == "H4":
            out.append(f"- H4: parameter ratio {c['param_ratio']:.3f}, MAC ratio {c['mac_ratio']:.3f} "
                       f"(range {c['range'][0]}–{c['range'][1]}) => {'supported' if c['supported'] else 'not supported'}")
    out.append("\n## All paired comparisons (descriptive)\n")
    for r in res["pairs"]:
        out.append(f"- {r['label']} [{pf(r)}; lower in {r['a_lower']} of {r['n']} seeds]")
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", nargs="?", default="results")
    ap.add_argument("--json", default=None, help="also write all numbers to this JSON file")
    ap.add_argument("--eff-seeds", default=None,
                    help="comma-separated seeds used for the cost table (Table 9) and the time ratios, e.g. 0,1,2 when the other seeds ran on another machine")
    a = ap.parse_args()
    runs = load(a.dir)
    if not runs:
        print("no *_s<seed>.json files found in", a.dir); return
    res = analyse(runs, {int(x) for x in a.eff_seeds.split(",")} if a.eff_seeds else None)
    print(to_markdown(res, a.dir))
    if a.json:
        Path(a.json).write_text(json.dumps(clean(res), indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
