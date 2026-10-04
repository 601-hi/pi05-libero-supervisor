#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/gpufree-data/vla-workspace/openpi
TRACE=/root/gpufree-data/libero-traces/abnormal_expert_v1
LOG=/root/gpufree-data/supervisor-results/logs
PY="$ROOT/examples/libero/.venv/bin/python"
run_one() {
  local condition="$1" scale="$2" steps="$3"
  echo "TEST_BATCH_START condition=$condition $(date --iso-8601=seconds)"
  cd "$ROOT"
  LIBERO_CONFIG_PATH=/root/gpufree-data/libero-config PYTHONPATH="$ROOT/third_party/libero" \
    "$PY" examples/libero/monitoring_main.py \
      --args.task-suite-name libero_spatial --args.num-trials-per-task 2 \
      --args.replan-steps 5 --args.seed 21 --args.sampling-noise-seed 2026090321 \
      --args.disturbance-start-mode random --args.disturbance-random-start-min 20 \
      --args.disturbance-random-start-max 90 --args.disturbance-num-steps "$steps" \
      --args.translation-action-scale "$scale" \
      --args.trace-out-path "$TRACE/test_seed21_${condition}.jsonl" \
      > "$LOG/test_seed21_${condition}.log" 2>&1
  echo "TEST_BATCH_DONE condition=$condition $(date --iso-8601=seconds)"
}
run_one normal 1.0 0
run_one scale025 0.25 10
run_one scale050 0.5 10
run_one scale075 0.75 10
echo "ALL_TEST_BATCHES_DONE $(date --iso-8601=seconds)"
