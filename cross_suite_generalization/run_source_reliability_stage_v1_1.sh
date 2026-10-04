#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/gpufree-data/supervisor-tools-v1
DATA="$ROOT/outputs/remote_results/droid100_training_ready"
OUT="$ROOT/outputs/remote_results/reliability_sequence_v1"
models=()
for seed in 202609111 202609112 202609113 202609114 202609115; do
  test -s "$OUT/expert_${seed}.pt"
  models+=(--model "$OUT/expert_${seed}.pt")
done

python "$ROOT/cross_suite_generalization/fit_reliability_calibration.py" \
  --data-directory "$DATA" "${models[@]}" \
  --normal-lr-quantile .50 --abnormal-lr-quantile .50 \
  --output "$OUT/reliability_calibration_v1_1.json" \
  --sequence-scores-output "$OUT/source_sequence_tune_scores_v1_1.npz"

python "$ROOT/cross_suite_generalization/fit_source_sequence_parameters.py" \
  --calibration "$OUT/reliability_calibration_v1_1.json" \
  --scores "$OUT/source_sequence_tune_scores_v1_1.npz" \
  --output "$OUT/source_sequence_parameters_v1_1.json"

python -m unittest discover -s "$ROOT/cross_suite_generalization" \
  -p 'test_reliability_sequence_supervisor.py' -v
