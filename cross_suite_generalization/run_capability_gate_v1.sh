#!/usr/bin/env bash
set -u
ROOT=/root/gpufree-data/vla-workspace/openpi
OUT=/root/gpufree-data/libero-traces/cross_suite_pilot_v1
mkdir -p "$OUT" "$OUT/videos" "$OUT/visuals"
run_one() {
  suite="$1"
  task="$2"
  stem="gate_${suite}_task${task}_seed30_normal"
  echo "GATE_START suite=$suite task=$task $(date -Iseconds)"
  cd "$ROOT" || return 1
  LIBERO_CONFIG_PATH=/root/gpufree-data/libero-config \
  PYTHONPATH="$ROOT/third_party/libero" \
  examples/libero/.venv/bin/python examples/libero/monitoring_main.py \
    --args.task-suite-name "$suite" \
    --args.task-id "$task" \
    --args.num-trials-per-task 1 \
    --args.replan-steps 5 \
    --args.seed 30 \
    --args.sampling-noise-seed 2026090530 \
    --args.disturbance-start-mode random \
    --args.disturbance-random-start-min 30 \
    --args.disturbance-random-start-max 80 \
    --args.disturbance-num-steps 0 \
    --args.translation-action-scale 1.0 \
    --args.trace-out-path "$OUT/${stem}.jsonl" \
    --args.video-out-path "$OUT/videos/$stem" \
    --args.visual-out-path "$OUT/visuals/$stem" \
    --args.visual-stride 1 \
    > "$OUT/${stem}.log" 2>&1
  status=$?
  echo "GATE_DONE suite=$suite task=$task status=$status $(date -Iseconds)"
  return 0
}
run_one libero_object 0
run_one libero_goal 0
run_one libero_90 0
echo "ALL_GATES_DONE $(date -Iseconds)"
