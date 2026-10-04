#!/usr/bin/env bash
# Requires a running pi05_libero policy server. Candidate pilot; does not overwrite frozen baselines.
set -u
ROOT=/root/gpufree-data/vla-workspace/openpi
OUT=/root/gpufree-data/libero-traces/object_consequence_seed32_v1
EVAL=examples/libero/monitoring_main.object_level_candidate_20260907.py
mkdir -p "$OUT/videos" "$OUT/visuals"

run_one() {
  suite="$1"; task="$2"; condition="$3"
  stem="${suite}_task${task}_seed32_${condition}"
  extra=()
  if [ "$condition" = "force_open_at_close" ]; then
    extra+=(--args.disturbance-start-mode gripper_close_event)
    extra+=(--args.disturbance-num-steps 15)
    extra+=(--args.translation-action-scale 1.0)
    extra+=(--args.gripper-disturbance-mode force_open)
    extra+=(--args.gripper-event-close-threshold 0.5)
    extra+=(--args.gripper-event-consecutive-steps 1)
  fi
  echo "OBJECT_PILOT_START suite=$suite task=$task condition=$condition $(date -Iseconds)"
  cd "$ROOT" || return 1
  LIBERO_CONFIG_PATH=/root/gpufree-data/libero-config PYTHONPATH="$ROOT/third_party/libero" \
  examples/libero/.venv/bin/python "$EVAL" \
    --args.task-suite-name "$suite" --args.task-id "$task" --args.num-trials-per-task 1 \
    --args.replan-steps 5 --args.seed 32 --args.sampling-noise-seed 2026090732 \
    --args.trace-out-path "$OUT/${stem}.jsonl" \
    --args.video-out-path "$OUT/videos/$stem" \
    --args.visual-out-path "$OUT/visuals/$stem" --args.visual-stride 1 \
    --args.record-oracle-object-state --args.record-kinematic-safety \
    "${extra[@]}" > "$OUT/${stem}.log" 2>&1
  status=$?
  echo "OBJECT_PILOT_DONE suite=$suite task=$task condition=$condition status=$status $(date -Iseconds)"
}

for spec in "libero_object 0" "libero_goal 2"; do
  set -- $spec
  run_one "$1" "$2" normal
  run_one "$1" "$2" force_open_at_close
done
echo "OBJECT_CONSEQUENCE_PILOT_DONE $(date -Iseconds)"
