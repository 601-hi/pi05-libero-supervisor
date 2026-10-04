#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/gpufree-data/vla-workspace/openpi
OUT=/root/gpufree-data/libero-traces/natural_validation_seed35_v1
mkdir -p "$OUT/traces" "$OUT/visual_sidecars" "$OUT/videos" "$OUT/logs"
ss -ltn 'sport = :8000' | grep -q LISTEN || { echo POLICY_SERVER_NOT_LISTENING; exit 3; }

run_one() {
  local task="$1" trials="$2"
  local stem="libero_90_task${task}_seed35_natural_${trials}ep"
  local trace="$OUT/traces/$stem.jsonl"
  test ! -e "$trace" || { echo "REFUSE_OVERWRITE $trace"; exit 2; }
  echo "VALIDATION_BATCH_START task=$task trials=$trials $(date -Iseconds)"
  cd "$ROOT"
  PYTHONUTF8=1 LIBERO_CONFIG_PATH=/root/gpufree-data/libero-config \
  PYTHONPATH="$ROOT/third_party/libero" \
    examples/libero/.venv/bin/python examples/libero/monitoring_main.py \
      --args.task-suite-name libero_90 --args.task-id "$task" \
      --args.num-trials-per-task "$trials" --args.replan-steps 5 \
      --args.seed 35 --args.sampling-noise-seed 2026091835 \
      --args.disturbance-num-steps 0 --args.translation-action-scale 1.0 \
      --args.trace-out-path "$trace" --args.video-out-path "$OUT/videos" \
      --args.visual-out-path "$OUT/visual_sidecars" --args.visual-stride 1 \
      >"$OUT/logs/$stem.log" 2>&1
  echo "VALIDATION_BATCH_DONE task=$task $(date -Iseconds)"
}

run_one 0 5
run_one 29 5
echo "NATURAL_VALIDATION_SEED35_COMPLETE $(date -Iseconds)"
