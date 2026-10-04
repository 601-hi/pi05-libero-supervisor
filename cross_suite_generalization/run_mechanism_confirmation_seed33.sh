#!/usr/bin/env bash
set -euo pipefail

# GPU collection only. Start the pi0.5 policy server separately on port 8000.
cd /root/gpufree-data/vla-workspace/openpi
PY=examples/libero/.venv/bin/python
OUT=/root/gpufree-data/libero-traces/mechanism_confirmation_seed33
VISUAL="$OUT/visual_sidecars"
VIDEO="$OUT/videos"
mkdir -p "$OUT" "$VISUAL" "$VIDEO"

ss -ltn 'sport = :8000' | grep -q LISTEN || {
  echo "POLICY_SERVER_NOT_LISTENING_ON_8000"
  exit 3
}

run_one() {
  local task="$1" condition="$2" trials="$3" visual="$4"
  local scale steps
  case "$condition" in
    normal) scale=1.0; steps=0 ;;
    scale025) scale=0.25; steps=10 ;;
    scale050) scale=0.50; steps=10 ;;
    scale075) scale=0.75; steps=10 ;;
    *) echo "UNKNOWN_CONDITION $condition"; exit 4 ;;
  esac
  local trace="$OUT/libero_10_task${task}_seed33_${condition}.jsonl"
  test ! -e "$trace" || { echo "REFUSE_OVERWRITE $trace"; exit 2; }
  local visual_args=()
  if [[ "$visual" == "yes" ]]; then
    visual_args=(--args.visual-out-path "$VISUAL" --args.visual-stride 1)
  fi
  echo "COLLECTION_START task=$task condition=$condition trials=$trials"
  LIBERO_CONFIG_PATH=/root/gpufree-data/libero-config \
  PYTHONPATH=/root/gpufree-data/vla-workspace/openpi/third_party/libero \
    "$PY" examples/libero/monitoring_main.py \
      --args.task-suite-name libero_10 --args.task-id "$task" \
      --args.num-trials-per-task "$trials" --args.replan-steps 5 \
      --args.seed 33 --args.sampling-noise-seed 2026091133 \
      --args.disturbance-start-mode random \
      --args.disturbance-random-start-min 30 --args.disturbance-random-start-max 80 \
      --args.disturbance-num-steps "$steps" --args.translation-action-scale "$scale" \
      --args.trace-out-path "$trace" --args.video-out-path "$VIDEO" \
      "${visual_args[@]}"
  echo "COLLECTION_DONE task=$task condition=$condition"
}

# Device calibration: normal-only, tasks already opened during development.
for task in 1 3 5 7 9; do
  run_one "$task" normal 4 no
done

# Confirmation: previously unseen even LIBERO-10 tasks. Visual sidecars are
# recorded but forbidden to the mechanism-supervisor confirmation evaluator.
for task in 0 2 4 6 8; do
  for condition in normal scale025 scale050 scale075; do
    run_one "$task" "$condition" 2 yes
  done
done

echo "MECHANISM_CONFIRMATION_COLLECTION_COMPLETE"
