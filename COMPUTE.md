# Running the experiments (where, what, how long)

## 1. Where it runs
A GPU is needed (a CPU is enough for the smoke test only). The paper's runs were made on two free services with the
same code files (the SHA-256 prefixes of `train.py`, `model.py`, `data.py`, `metrics.py`, `posthoc.py` and `run_plan.sh`
are identical in `env.json` and `env_kaggle.json`) and the same data file:

| | Colab (`MT-STGNN_Colab.ipynb`) | Kaggle (`MT-STGNN_Kaggle.ipynb`) |
|---|---|---|
| GPU | one T4 (15 GB) | two T4 (15 GB each), one training per GPU |
| Python / PyTorch | 3.13.16 / 2.14.1 (CUDA 13.0, cuDNN 9.28) | 3.13.16 / 2.11.0 (CUDA 13.0, cuDNN 9.27) |
| NumPy / pandas | 2.4.6 / 2.2.3 | 2.1.3 / 2.3.3 |
| What ran there | seeds 0–2 of R, C, MT-F (β = 1), MT-U, MT-P | the β grid, the tuned MT-F, seeds 3–5 of the core variants, the sensitivity and ablation runs, the 100-epoch check, the platform check |

The Kaggle runs come from three sessions of the same notebook (section 2). Their environment records (`kaggle_sessions/` in the
results package) agree in the library versions, the GPU model, the driver, the CPU and the hashes of the source files, and the
dataset statistics agree; a Kaggle session of this kind is therefore one machine for the purpose of the paper.

Because the two PyTorch builds differ, a comparison between a Colab run and a Kaggle run can contain a machine effect.
The paper therefore also reports the paired gaps on the seeds that all ran on Kaggle, and a platform check (R and
MT-F with β = 1, seeds 0–2, repeated on Kaggle). Training time per epoch (Table 9) comes from the Colab seeds only: the
Kaggle machine trains two models at once.

Measured on the Colab T4 (METR-LA, 207 sensors, batch 64, about 24 k training windows per epoch, `--amp`, mean of seeds 0–2):

| Variant | R (reg) | C (cls) | MT-F | MT-U | MT-P (PCGrad) |
|---|---|---|---|---|---|
| seconds per epoch | 18.9 | 19.2 | 20.5 | 20.7 | 30.7 |

In an earlier test on the same GPU the regression model needed 63–68 s per epoch in float32. `--amp` (float16 encoder;
heads and losses in float32) is therefore used for all runs; in a 100-batch test with the same seed it changed the
validation loss (0.5628 vs 0.5606) and the test MAE (4.108 vs 4.094 mph) by less than 0.5%.
Batch 128 was not faster on a T4, and running 2–3 trainings at once on one GPU gave no extra throughput (the GPU is
saturated), so the plan uses batch 64 and `AMP=1`, and the Kaggle queue puts one training on each of its two GPUs.

## 2. The plan
`run_plan.sh` (one GPU) and `kaggle_run.py` (several GPUs) issue the same `train.py` and `posthoc.py` commands. Both
are resumable: a run is skipped when its JSON already exists, so an interrupted job is simply started again.
Prediction arrays (`.npz`) are kept for seed 0 only (disk).

| Stage | Command | Runs | Where |
|---|---|---|---|
| smoke | `bash run_plan.sh smoke` | 5 tiny runs on synthetic data | both |
| core | `bash run_plan.sh core metr-la data/METR-LA` | R, C, MT-F (β = 1), MT-U, MT-P × `SEEDS` | Colab: seeds 0–2; Kaggle (version A): seeds 3–5 |
| grid | `GRID="0.25 0.5 2 4" SEEDS="0 1 2" bash run_plan.sh grid metr-la data/METR-LA` | MT-F with β ≠ 1, seeds 0–2 (12 runs) | Kaggle (session 1) |
| tuned | `SEEDS="3 4 5" GRID=<best β> bash run_plan.sh grid metr-la data/METR-LA` | the selected β on seeds 3–5 (3 runs) | Kaggle (version A) |
| sens | `SEEDS=0 bash run_plan.sh sens metr-la data/METR-LA` | Table 11: label variants for C and MT-P, two ablations (10 runs) | Kaggle (version B) |
| e100 | five core variants, seed 0, up to 100 epochs | 5 runs | Kaggle (version B) |
| xcheck | R and MT-F (β = 1), seeds 0–2 | 6 runs | Kaggle (version B) |
| summary | `bash run_plan.sh summary` | `aggregate.py` → `summary.md` | anywhere |

The β is chosen by the mean best validation loss over seeds 0–2 (the same criterion as early stopping). On the
Kaggle machine the whole queue is one call (about 9 hours on two T4 GPUs):

    python kaggle_run.py --stages grid,tuned,core,sens,e100,xcheck --out results --b1-val b1_val.json

### How the Kaggle runs of the paper were split into sessions
The notebook was first run as an interactive session (session 1, `STAGES = "grid,tuned,core,sens,e100"`). The grid
finished, and its result (selected β = 0.5; the validation losses are in `kaggle_sessions/session1_status_kaggle.json`)
was exported. Kaggle ends an interactive session after about an hour without a running cell and deletes its working
directory; this happened to session 1 while the later stages were running, before their results had been exported.
None of the lost runs was used. The stages were run again from the start, with the same commands, as two committed
versions of the notebook ("Save & Run All", which runs on Kaggle's servers and keeps its output):

* version A (session 2), `STAGES = "tuned,core"`: the tuned MT-F and the core variants on seeds 3–5 (18 runs);
* version B (session 3), `STAGES = "xcheck,sens,e100"`: the platform check (6 runs), the sensitivity and ablation runs (10 runs) and the 100-epoch check (5 runs).

Both versions wrote the β decision of session 1 into the status file before the queue started (`PRESEED` in cell 8 of the
notebook), so the grid was not repeated and the tuned β is the one that was selected on seeds 0–2 in session 1.

Time of the Kaggle runs (two trainings at once, one per T4; minutes of one run, 50 epochs unless stated):

| Stage | Session | Runs | Minutes per run: median (range) | Wall-clock of the stage (h) |
|---|---|---|---|---|
| grid | 1 | 12 | 19.4 (19.1–19.7) | 2.0 |
| tuned | 2 | 3 | 17.8 (17.7–17.8) | 0.6 |
| core | 2 | 15 | 17.7 (16.1–25.8) | 2.4 |
| sens | 3 | 10 | 23.8 (14.6–27.8) | 2.0 |
| e100 (100 epochs) | 3 | 5 | 22.1 (17.9–41.2) | 1.1 |
| xcheck | 3 | 6 | 17.5 (16.4–19.1) | 0.9 |

session 1 (grid): about 2.0 h from the start of its first run to the end of its last one; session 2 (tuned, core): about 3.0 h from the start of its first run to the end of its last one; session 3 (xcheck, sens, e100): about 3.9 h from the start of its first run to the end of its last one; 8.9 h in total (a stage ends when both GPUs have finished, so one GPU can be idle at the end of a stage).

Core stage, minutes per run on the Kaggle T4 (seeds 3–5, median):

| R | C | MT-F (β = 1) | MT-U | MT-P |
|---|---|---|---|---|
| 16.3 | 16.4 | 17.8 | 17.9 | 25.8 |

A run's minutes include the evaluation and the latency measurement, and two trainings ran at the same time on the host, so these minutes are not comparable with the seconds per epoch of Table 9 (Colab, one training at a time).

## 3. Data
One folder per dataset: `metr-la.h5` + `adj_mx.pkl` (METR-LA). The `.h5` file comes from the DCRNN authors' Google
Drive folder (linked in github.com/liyaguang/DCRNN). The notebooks rebuild the METR-LA adjacency from two small data files of
that repository with the DCRNN rule. (`pems-bay.h5` + `adj_mx_bay.pkl` and the `pems` stage of `run_plan.sh` are
included for a replication, which was not run for the paper.)

## 4. What to keep or send back
Every run's JSON with the gradient-cosine log thinned to every 20th step is in `results_thin.zip` (Colab: in
`MyDrive/mtstgnn/export`; Kaggle: in `/kaggle/working/export`, together with `results_e100_thin.zip`,
`results_xcheck_thin.zip`, `env_kaggle.json`, `ds_metr_kaggle.json` and `status_kaggle.json`). Kaggle deletes the working
directory when a session ends: a committed version keeps its output (Output tab of the notebook page, once the version has
finished), an interactive session does not, so download these files before closing it and do not rely on an interactive
session for the long queue. The run files of several sessions are merged by unpacking them into one folder (a run file
name is unique); give the environment records of the sessions different names. Do not edit numbers by hand: `aggregate.py` regenerates every
table and the H1–H4 checks from the run files:

    python aggregate.py <dir with the run JSONs> --json out.json --eff-seeds 0,1,2 > summary.md

(`--eff-seeds` restricts the cost table to the seeds that ran on the Colab T4.)

## 5. Known limitations of the scripts
* `posthoc.py` truncates the persistence window of the R→thr baseline at the first forecast step (the `.npz` holds no observed past).
* Gradient cosines are logged every 20th step (every step for PCGrad); `aggregate.py` uses every 20th step for all variants.
* Best β for MT-F is picked by the mean best validation loss (unweighted sum of the two task losses), the same criterion as early stopping.
* Training is capped at 50 epochs; the 100-epoch check on seed 0 (the five core variants) is meant to quantify the effect of this cap.
* Bootstrap intervals use seed 0 only (predictions are kept for seed 0) and the Colab runs of MT-F (β = 1), MT-U, MT-P, R and C, so they measure test-set sampling uncertainty, not training randomness; the tuned MT-F has none because its seed-0 predictions were produced on the other machine.
* With three seeds the exact paired Wilcoxon test cannot reach p < 0.05 (the smallest two-sided p is 0.25), with six seeds the smallest is 0.031; the paper reports tests from six seeds on.
