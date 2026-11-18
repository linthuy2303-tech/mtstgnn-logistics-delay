"""Merge results/*.json into one markdown table (+ a 'two separate models' row).
Usage: python summarize.py results/ [reg_run_name cls_run_name]
"""
import json
import sys
from pathlib import Path

d = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
runs = {p.stem: json.loads(p.read_text()) for p in sorted(d.glob("*.json"))}


def row(name, r):
    t, e = r["test"], r["efficiency"]
    g = lambda k, f: (f"{t[k]['avg'][f]:.3f}" if k in t else "-")
    rec = f"{t['cls']['avg']['recall'][2]:.3f}" if "cls" in t else "-"
    return (f"| {name} | {g('reg','mae')} | {g('reg','rmse')} | {g('cls','macro_f1')} | {rec} | "
            f"{g('cls','pr_auc_level2')} | {e['params']:,} | {e['train_s_per_epoch']:.1f} | "
            f"{e['latency_ms_batch1']:.2f} |")


print("| run | MAE | RMSE | macro-F1 | recall L2 | PR-AUC L2 | params | s/epoch | ms (batch=1) |")
print("|---|---|---|---|---|---|---|---|---|")
for n, r in runs.items():
    print(row(n, r))
if len(sys.argv) == 4 and sys.argv[2] in runs and sys.argv[3] in runs:
    a, b = runs[sys.argv[2]], runs[sys.argv[3]]
    ea, eb = a["efficiency"], b["efficiency"]
    print(f"| two separate models ({sys.argv[2]} + {sys.argv[3]}) | {a['test']['reg']['avg']['mae']:.3f} | "
          f"{a['test']['reg']['avg']['rmse']:.3f} | {b['test']['cls']['avg']['macro_f1']:.3f} | "
          f"{b['test']['cls']['avg']['recall'][2]:.3f} | {b['test']['cls']['avg'].get('pr_auc_level2', float('nan')):.3f} | "
          f"{ea['params'] + eb['params']:,} | {ea['train_s_per_epoch'] + eb['train_s_per_epoch']:.1f} | "
          f"{ea['latency_ms_batch1'] + eb['latency_ms_batch1']:.2f} |")
