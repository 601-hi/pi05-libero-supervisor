#!/usr/bin/env bash
set -euo pipefail
cd /root/gpufree-data/vla-workspace/openpi
PY=examples/libero/.venv/bin/python
OUT=/root/gpufree-data/libero-traces/libero10_sealed_transfer_v1
mkdir -p "$OUT"
for task in 1 3 5 7 9; do
  for condition in normal scale025 scale050 scale075; do
    case "$condition" in normal) scale=1.0; steps=0;; scale025) scale=0.25; steps=10;; scale050) scale=0.5; steps=10;; scale075) scale=0.75; steps=10;; esac
    trace="$OUT/libero_10_task${task}_seed32_${condition}.jsonl"
    test ! -e "$trace" || { echo "REFUSE_OVERWRITE $trace"; exit 2; }
    LIBERO_CONFIG_PATH=/root/gpufree-data/libero-config PYTHONPATH=/root/gpufree-data/vla-workspace/openpi/third_party/libero \
      "$PY" examples/libero/monitoring_main.py --args.task-suite-name libero_10 --args.task-id "$task" \
      --args.num-trials-per-task 2 --args.replan-steps 5 --args.seed 32 --args.sampling-noise-seed 2026090532 \
      --args.disturbance-start-mode random --args.disturbance-random-start-min 30 --args.disturbance-random-start-max 80 \
      --args.disturbance-num-steps "$steps" --args.translation-action-scale "$scale" --args.trace-out-path "$trace"
  done
done
echo "SEALED_COLLECTION_COMPLETE"
