#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/gpufree-data/vla-workspace/openpi
TRACE=/root/gpufree-data/libero-traces/abnormal_expert_v1
LOG=/root/gpufree-data/supervisor-results/logs
PYTHONPATH_VALUE="$ROOT/third_party/libero"
PY="$ROOT/examples/libero/.venv/bin/python"

run_one() {
  local split="$1" seed="$2" condition="$3" scale="$4" steps="$5"
  local noise_seed=$((2026090300 + seed))
  echo "BATCH_START split=$split seed=$seed condition=$condition $(date --iso-8601=seconds)"
  cd "$ROOT"
  LIBERO_CONFIG_PATH=/root/gpufree-data/libero-config PYTHONPATH="$PYTHONPATH_VALUE" \
    "$PY" examples/libero/monitoring_main.py \
      --args.task-suite-name libero_spatial \
      --args.num-trials-per-task 1 \
      --args.replan-steps 5 \
      --args.seed "$seed" \
      --args.sampling-noise-seed "$noise_seed" \
      --args.disturbance-start-mode random \
      --args.disturbance-random-start-min 20 \
      --args.disturbance-random-start-max 90 \
      --args.disturbance-num-steps "$steps" \
      --args.translation-action-scale "$scale" \
      --args.trace-out-path "$TRACE/${split}_seed${seed}_${condition}.jsonl" \
      > "$LOG/${split}_seed${seed}_${condition}.log" 2>&1
  echo "BATCH_DONE split=$split seed=$seed condition=$condition $(date --iso-8601=seconds)"
}

mkdir -p "$TRACE" "$LOG"
for spec in "normal 1.0 0" "scale025 0.25 10" "scale050 0.5 10" "scale075 0.75 10"; do
  read -r condition scale steps <<< "$spec"
  run_one train 19 "$condition" "$scale" "$steps"
done
for spec in "normal 1.0 0" "scale025 0.25 10" "scale050 0.5 10" "scale075 0.75 10"; do
  read -r condition scale steps <<< "$spec"
  run_one calibration 20 "$condition" "$scale" "$steps"
done
echo "ALL_BATCHES_DONE $(date --iso-8601=seconds)"
