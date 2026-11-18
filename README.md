# Multi-task ST-GNN: ETA delay regression + disruption early warning

Code of the paper *Multi-Task Spatio-Temporal Graph Neural Network for Joint ETA Delay Prediction and Disruption Risk
Classification in Logistics Networks: Accuracy and Efficiency Analysis*. One shared graph encoder, two heads: speed
(delay) regression and a three-level disruption risk. The experiments of the paper ran on two free GPU services:
seeds 0–2 of the five core variants on one Colab T4 (notebook `MT-STGNN_Colab.ipynb`), everything else on a Kaggle
notebook with two T4 GPUs (notebook `MT-STGNN_Kaggle.ipynb`, job queue `kaggle_run.py`; three sessions, see `COMPUTE.md`), with
the same code files on both machines. On any other CUDA GPU use `run_plan.sh`. `COMPUTE.md` has the commands, the plan, the timings and the
list of what ran where.

## Files
| File | What it does |
|---|---|
| `data.py` | loading of DCRNN-format data, normalisation, chronological split, 3-level risk labels |
| `model.py` | shared encoder (gated temporal convolutions + graph convolution on forward/backward random-walk supports and an adaptive adjacency) and the two heads |
| `train.py` | training (fixed weights, uncertainty weighting, PCGrad), metrics, efficiency (parameters, MACs, time per epoch, latency), `--amp`, `--log_grad_cos`, `--save_preds` |
| `metrics.py` | regression and classification metrics |
| `run_plan.sh` | staged, resumable plan for one GPU: `smoke`, `core`, `grid`, `sens`, `pems`, `summary` |
| `kaggle_run.py` | the same runs as a queue for a machine with several GPUs (one training per GPU): stages `grid`, `tuned`, `core`, `sens`, `e100`, `xcheck`; picks the tuned β by the rule of `aggregate.py` and exports thin result zips after every run |
| `posthoc.py` | extra per-run metrics from the saved predictions (average MAE, delay MAE, errors by risk level, the R→thr baseline) |
| `aggregate.py` | mean ± SD over seeds, paired Wilcoxon tests and the four pre-specified checks H1–H4 → markdown and JSON; `--eff-seeds` restricts the cost table to the seeds of one machine |
| `bootstrap.py` | day-block bootstrap of paired differences on the seed-0 test predictions |
| `datastats.py` | dataset statistics (sensors, steps, missing share, label shares, split) |
| `eval_warning.py`, `plot_reliability.py` | calibration, event lead time and conformal intervals; for a separate study, not used in the tables of the paper |
| `summarize.py`, `run_all.sh` | a quick one-seed table (helper) |

## Setup
    pip install torch numpy pandas tables scipy        # matplotlib only for plot_reliability.py
DCRNN-format files: `metr-la.h5` + `adj_mx.pkl` (METR-LA), `pems-bay.h5` + `adj_mx_bay.pkl` (PEMS-BAY), source
github.com/liyaguang/DCRNN (the `.h5` files are linked from its README). The notebooks fetch them themselves.
The paper uses METR-LA only (`metr-la.h5`, SHA-256 64784b76…).

## Smoke test (no data needed)
    bash run_plan.sh smoke

## Experiments
    bash run_plan.sh core metr-la data/METR-LA          # R, C, MT-F (beta = 1), MT-U, MT-P
    bash run_plan.sh summary                            # tables and H1-H4 checks -> results/summary.md
On a machine with two GPUs:
    python kaggle_run.py --stages grid,tuned,core,sens,e100,xcheck --out results --b1-val b1_val.json
(`b1_val.json` holds the validation losses of MT-F with β = 1 on seeds 0–2, which come from the Colab runs.) On Kaggle run the
notebook as a committed version ("Save & Run All"): an interactive session is ended after about an hour without a running cell
and its results are lost (`COMPUTE.md`, section 2).
See `COMPUTE.md` for the full plan (seeds, β grid, sensitivity runs), for what was run where and for the running times.
`--log_grad_cos` stores the cosine between the two task gradients on the shared encoder (`grad_cos` in each run's
JSON): negative values mean a real gradient conflict.

## Results behind the paper
`mtstgnn_results.zip` (not included in this repository) holds what the tables of the paper are computed from:
- `metr_raw/`: one JSON per training run on METR-LA (`<variant>_s<seed>.json`) and the `_post.json` that `posthoc.py` wrote for it. Seeds 0–2 of R, C, MT-F (β = 1), MT-U and MT-P come from the Colab T4; every other run (β grid, tuned MT-F, seeds 3–5, sensitivity and ablation runs) comes from the Kaggle notebook (`PROVENANCE.csv` names the machine, and for Kaggle the session, of each run file);
- `e100_raw/` (the 100-epoch check, seed 0) and `xcheck_raw/` (R and MT-F with β = 1 repeated on Kaggle for seeds 0–2);
- `env.json` and `env_kaggle.json` (library versions and a hash of every source file used on each machine; the Kaggle record is that of session 1), `ds_metr.json` and `ds_metr_kaggle.json` (dataset statistics computed on each machine);
- `kaggle_sessions/` (the environment record, the dataset statistics and the status file with the minutes of every run, for each of the three Kaggle sessions);
- the aggregated results (`metr.json`, `summary.md`, `e100.json`, `xcheck_metr.json`, `bootstrap_metr.json`), `config.json` (switches used to build the manuscript text) and `README.txt`.

The tables are regenerated from the run files (`aggregate.py` is in this package):

    python aggregate.py metr_raw --json metr.json --eff-seeds 0,1,2 > summary.md
    python aggregate.py e100_raw --json e100.json > /dev/null
    python aggregate.py xcheck_raw --json xcheck_metr.json > /dev/null

`bootstrap_metr.json` comes from `bootstrap.py` on the seed-0 test predictions of the Colab runs (the prediction arrays are not part of the thin result set). Every table of the paper is read from these files; nothing is typed in by hand. Because `run_plan.sh` skips a run whose JSON exists, do not unpack the run files into the `OUT` directory of a new experiment unless you want to continue from them.

## Design decisions to check
- The regression target is speed (L1 loss on the normalised scale, metrics in mph). Delay in minutes per mile is
  derived as |60 / v̂ − 60 / v| in `posthoc.py`.
- Labels: v_free is the 85th percentile of the 00:00–05:00 speeds on the training split; ratio thresholds 0.70 / 0.40;
  `--persist_k 3` requires the condition to hold on three consecutive steps (1 gives plain thresholds). METR-LA has no
  incident labels, so level 2 means severe congestion by speed, not an accident.
- Report macro-F1, recall and PR-AUC of level 2; accuracy is dominated by the safe class.
- Mixed precision (`--amp`): the encoder runs in float16 with a static loss scale (1024, halved and the step skipped
  on a non-finite gradient norm, doubled after 500 good steps); the heads and all losses stay in float32. The number
  of skipped steps is stored in each run (`amp_skipped_steps`). Latency is always measured in float32.
- Reproducibility: seeds fix the initial weights (the same seed gives the same encoder in every variant) but cuDNN
  autotuning is on, so runs are not bitwise reproducible. Each machine writes an `env*.json` with the library versions
  and a hash of every source file used; the hashes of `train.py`, `model.py`, `data.py`, `metrics.py`, `posthoc.py` and
  `run_plan.sh` are identical on the two machines and in all Kaggle sessions.
- Two machines: the PyTorch builds differ (see `COMPUTE.md`), so a difference between a Colab run and a Kaggle run
  can contain a machine effect. The paper reports the paired gaps on the same-machine seeds and a platform check
  (`xcheck`: R and MT-F with β = 1 on seeds 0–2 repeated on Kaggle). Time per epoch (Table 9) uses the Colab seeds only,
  because the Kaggle machine trains two models at once.

## Status of the code
Run for the paper: the core stage (five variants, six seeds); the β grid on seeds 0–2 and the selected β on seeds 3–5; the sensitivity and ablation runs; the 100-epoch check on seed 0; the platform check; the bootstrap (`bootstrap.py`). The PEMS-BAY replication (`pems` stage, `c10`/`c11` cells of the Colab notebook) is implemented but was **not run**; the paper says so (Sections 5.3 and 9). The numbers in the paper come from `aggregate.py` applied to the run files; nothing is typed in by hand. The Kaggle notebook that ran the queue embedded a copy of `aggregate.py` from before the option `--eff-seeds` was added; apart from that option, which only restricts the cost table, the code of the notebook, the training code and the rule that selects β are the files of this package. The numpy-only parts (`data.py`, `metrics.py`, `posthoc.py`, `aggregate.py`, `bootstrap.py`, `eval_warning.py`) were additionally tested on synthetic arrays.
