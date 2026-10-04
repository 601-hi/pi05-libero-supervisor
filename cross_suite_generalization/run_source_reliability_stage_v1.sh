#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/gpufree-data/supervisor-tools-v1
DATA="$ROOT/outputs/remote_results/droid100_training_ready"
OUT="$ROOT/outputs/remote_results/reliability_sequence_v1"
mkdir -p "$OUT"

for seed in 202609111 202609112 202609113 202609114 202609115; do
  python "$ROOT/cross_suite_generalization/relational_dual_expert.py" \
    --data-directory "$DATA" --output "$OUT/expert_${seed}.pt" \
    --seed "$seed" --threads 2 --components 8 --feature-set progress_only \
    --margin 2.0 --margin-weight 1.0 --scales .25 .5 .75
done

models=()
for seed in 202609111 202609112 202609113 202609114 202609115; do
  models+=(--model "$OUT/expert_${seed}.pt")
done

python "$ROOT/cross_suite_generalization/fit_reliability_calibration.py" \
  --data-directory "$DATA" "${models[@]}" \
  --output "$OUT/reliability_calibration.json" \
  --sequence-scores-output "$OUT/source_sequence_tune_scores.npz"

python "$ROOT/cross_suite_generalization/fit_source_sequence_parameters.py" \
  --calibration "$OUT/reliability_calibration.json" \
  --scores "$OUT/source_sequence_tune_scores.npz" \
  --output "$OUT/source_sequence_parameters.json"

python -m unittest discover -s "$ROOT/cross_suite_generalization" \
  -p 'test_reliability_sequence_supervisor.py' -v
