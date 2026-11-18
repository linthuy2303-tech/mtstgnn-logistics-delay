#!/usr/bin/env bash
# Staged, resumable experiment plan for the paper (Tables 5-12).
#
#   bash run_plan.sh smoke                           # 5 tiny runs on synthetic data: checks the whole pipeline
#   bash run_plan.sh core  metr-la data/METR-LA      # R, C, MT-F (beta=1), MT-U, MT-P  x 6 seeds        = 30 runs
#   bash run_plan.sh grid  metr-la data/METR-LA      # MT-F beta grid {0.25,0.5,2,4}    x 6 seeds        = 24 runs
#   bash run_plan.sh sens  metr-la data/METR-LA      # label variants (C and MT-P) + 2 ablations x 6 seeds = 60 runs
#   OUT=results_pems BETA=<best beta> bash run_plan.sh pems pems-bay data/PEMS-BAY   # R, C, MT-F, MT-P x 6 seeds = 24 runs
#   bash run_plan.sh summary                         # posthoc check + aggregate.py -> results/summary.md
#
# Environment: SEEDS="0 1 2 3 4 5"  EPOCHS=50  OUT=results  BETA=1 (beta used by the 'pems' stage)  GRID="0.25 0.5 2 4"
#              AMP=1 (float16 encoder)  PAR=2 (run 2 trainings at once on one GPU; each run's log goes to $OUT/logs/)
# Every run is skipped if its JSON already exists, so an interrupted job can simply be started again.
# Arrays (.npz) are kept only for seed 0 (disk); posthoc.py is run on every run that has speed forecasts.
set -u
STAGE=${1:?stage: smoke|core|grid|sens|pems|summary}
D=${2:-synthetic}; DIR=${3:-.}
SEEDS=${SEEDS:-"0 1 2 3 4 5"}; E=${EPOCHS:-50}; OUT=${OUT:-results}; BETA=${BETA:-1}
GRID=${GRID:-"0.25 0.5 2 4"}
PAR=${PAR:-1}                                  # number of training processes run at the same time on the GPU
AMPF=""; if [ "${AMP:-0}" = "1" ]; then AMPF="--amp"; fi   # AMP=1: float16 encoder (about 1.75x faster, same results to 3 digits)
mkdir -p "$OUT" "$OUT/logs"

_run() {  # _run <base> <seed> <extra train.py args...>
  local base=$1 seed=$2; shift 2
  python train.py --data "$D" --data_dir "$DIR" --epochs "$E" --log_grad_cos --seed "$seed" $AMPF \
      --save_preds --out "$base.json" "$@" || { echo "FAILED $base" >&2; return; }
  [ -f "$base.npz" ] && python posthoc.py --npz "$base.npz" --out "${base}_post.json"
  if [ "$seed" != "0" ]; then rm -f "$base.npz"; fi
}

run() {  # run <name> <seed> <extra train.py args...>
  local name=$1 seed=$2; shift 2
  local base="$OUT/${name}_s${seed}"
  if [ -f "$base.json" ]; then echo "skip $base"; return; fi
  if [ "$PAR" -le 1 ]; then _run "$base" "$seed" "$@"; return; fi
  while [ "$(jobs -rp | wc -l)" -ge "$PAR" ]; do wait -n; done
  ( _run "$base" "$seed" "$@" > "$OUT/logs/${name}_s${seed}.log" 2>&1; echo "finished ${name}_s${seed} $(date +%H:%M)" ) &
}

case "$STAGE" in
  smoke)
    OUT=results_smoke; mkdir -p $OUT; SEEDS="0"; E=2; D=synthetic; DIR=.
    for s in $SEEDS; do
      run reg  $s --mode reg  --max_batches 20
      run cls  $s --mode cls  --max_batches 20 --class_weight
      run mtf_b1 $s --mode multi --alpha 1 --beta 1 --max_batches 20 --class_weight
      run mtu  $s --mode multi --weighting uncertainty --max_batches 20 --class_weight
      run mtp  $s --mode multi --pcgrad --max_batches 20 --class_weight
    done
    python aggregate.py $OUT > $OUT/summary.md && head -40 $OUT/summary.md ;;
  core)
    for s in $SEEDS; do
      run reg    $s --mode reg
      run cls    $s --mode cls --class_weight
      run mtf_b1 $s --mode multi --alpha 1 --beta 1 --class_weight
      run mtu    $s --mode multi --weighting uncertainty --class_weight
      run mtp    $s --mode multi --pcgrad --class_weight
    done ;;
  grid)
    for b in $GRID; do for s in $SEEDS; do
      run "mtf_b$b" $s --mode multi --alpha 1 --beta "$b" --class_weight
    done; done ;;
  sens)
    for s in $SEEDS; do
      for cfg in "k1:--persist_k 1" "k6:--persist_k 6" "t60_30:--t_warn 0.60 --t_crit 0.30" "t80_50:--t_warn 0.80 --t_crit 0.50"; do
        tag=${cfg%%:*}; args=${cfg#*:}
        run "cls_$tag" $s --mode cls --class_weight $args
        run "mtp_$tag" $s --mode multi --pcgrad --class_weight $args
      done
      run mtp_noadj $s --mode multi --pcgrad --class_weight --no_adaptive
      run mtp_nocw  $s --mode multi --pcgrad
    done ;;
  pems)
    for s in $SEEDS; do
      run reg    $s --mode reg
      run cls    $s --mode cls --class_weight
      run "mtf_b$BETA" $s --mode multi --alpha 1 --beta "$BETA" --class_weight
      run mtp    $s --mode multi --pcgrad --class_weight
    done ;;
  summary)
    python aggregate.py "$OUT" > "$OUT/summary.md" && echo "wrote $OUT/summary.md" ;;
  *) echo "unknown stage $STAGE"; exit 1 ;;
esac
wait   # background runs (PAR > 1)
