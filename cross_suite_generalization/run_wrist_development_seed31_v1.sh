#!/usr/bin/env bash
# Pre-registered wrist-RGB development collection. Requires a running policy server.
set -u
ROOT=/root/gpufree-data/vla-workspace/openpi
OUT=/root/gpufree-data/libero-traces/wrist_development_seed31_v1
mkdir -p "$OUT/videos" "$OUT/visuals"

run_one() {
  suite="$1"; task="$2"; scale="$3"; nsteps="$4"; tag="$5"
  stem="${suite}_task${task}_seed31_${tag}"
  echo "WRIST_DEV_START suite=$suite task=$task tag=$tag $(date -Iseconds)"
  cd "$ROOT" || return 1
  LIBERO_CONFIG_PATH=/root/gpufree-data/libero-config PYTHONPATH="$ROOT/third_party/libero" \
  examples/libero/.venv/bin/python examples/libero/monitoring_main.py \
    --args.task-suite-name "$suite" --args.task-id "$task" --args.num-trials-per-task 1 \
    --args.replan-steps 5 --args.seed 31 --args.sampling-noise-seed 2026090531 \
    --args.disturbance-start-mode random --args.disturbance-random-start-min 30 \
    --args.disturbance-random-start-max 80 --args.disturbance-num-steps "$nsteps" \
    --args.translation-action-scale "$scale" --args.trace-out-path "$OUT/${stem}.jsonl" \
    --args.video-out-path "$OUT/videos/$stem" --args.visual-out-path "$OUT/visuals/$stem" \
    --args.visual-stride 1 > "$OUT/${stem}.log" 2>&1
  status=$?
  echo "WRIST_DEV_DONE suite=$suite task=$task tag=$tag status=$status $(date -Iseconds)"
}

# Tasks already passing at least one capability pilot. libero_10 remains sealed.
for spec in "libero_object 0" "libero_object 2" "libero_object 4" \
            "libero_goal 0" "libero_goal 2" "libero_goal 4" \
            "libero_90 9" "libero_90 19"; do
  set -- $spec
  run_one "$1" "$2" 1.0 0 normal
  run_one "$1" "$2" 0.75 10 scale075
  run_one "$1" "$2" 0.5 10 scale050
done
echo "WRIST_DEVELOPMENT_SEED31_DONE $(date -Iseconds)"
