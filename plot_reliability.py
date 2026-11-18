"""Reliability diagram of level 2 (raw vs temperature-scaled) from eval_warning.py output.
Usage: python plot_reliability.py results/multi_pcgrad_warning.json fig_reliability.png
"""
import json, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

res = json.load(open(sys.argv[1]))["calibration"]
fig, ax = plt.subplots(figsize=(3.2, 3.0), dpi=300)
ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="perfect")
for name, c in (("raw", "#a00000"), ("scaled", "#1f4e9c")):
    tab = res[name]["reliability_level2"]
    ax.plot([r["conf"] for r in tab], [r["freq"] for r in tab], "o-", ms=2.5, lw=0.9, color=c,
            label=f"{name} (ECE$_2$ = {res[name]['ece_level2']:.3f})")
ax.set_xlabel("predicted probability of level 2"); ax.set_ylabel("observed frequency")
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.legend(fontsize=6.5, frameon=False)
fig.tight_layout(); fig.savefig(sys.argv[2], bbox_inches="tight")
print("saved", sys.argv[2])
