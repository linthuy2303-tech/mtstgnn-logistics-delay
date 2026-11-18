#!/usr/bin/env bash
# Usage: bash run_all.sh metr-la data/METR-LA     (or: bash run_all.sh synthetic .)
set -e
D=${1:-synthetic}; DIR=${2:-.}; E=${EPOCHS:-50}
C="python train.py --data $D --data_dir $DIR --epochs $E --class_weight --log_grad_cos"
$C --mode reg   --out results/reg.json
$C --mode cls   --save_preds --out results/cls.json
$C --mode multi --out results/multi_fixed.json
$C --mode multi --weighting uncertainty --out results/multi_uncertainty.json
$C --mode multi --pcgrad --save_preds --out results/multi_pcgrad.json
python summarize.py results reg cls
# Paper 2: calibration, event-based lead time, conformal intervals
python eval_warning.py --preds results/cls.npz        --out results/cls_warning.json
python eval_warning.py --preds results/multi_pcgrad.npz --out results/multi_pcgrad_warning.json
