#!/usr/bin/env bash
set -euo pipefail

# Usage: bash run_expanded_natural_tier.sh development 36 2026091836
TIER=${1:?tier required}
SEED=${2:?environment seed required}
NOISE_SEED=${3:?policy sampling noise seed required}
case "$TIER" in
  development|model_selection|sealed_final_test|sealed_confirmation) ;;
  *) echo "INVALID_TIER $TIER"; exit 2 ;;
esac

ROOT=/root/gpufree-data/vla-workspace/openpi
OUT=/root/gpufree-data/libero-traces/expanded_visual_v1/${TIER}_seed${SEED}
MIN_FREE_KIB=$((8 * 1024 * 1024))
FREE_KIB=$(df --output=avail /root/gpufree-data | tail -1 | tr -d ' ')
(( FREE_KIB >= MIN_FREE_KIB )) || { echo "INSUFFICIENT_DISK_KIB $FREE_KIB"; exit 4; }
ss -ltn 'sport = :8000' | grep -q LISTEN || { echo POLICY_SERVER_NOT_LISTENING; exit 3; }
test ! -e "$OUT/COLLECTION_COMPLETE" || { echo "REFUSE_COMPLETED_TIER $OUT"; exit 5; }
mkdir -p "$OUT"/{traces,visual_sidecars,videos,logs}

run_one() {
  local suite=$1 task=$2 trials=$3
  local stem="${suite}_task${task}_seed${SEED}_natural_${trials}ep"
  local trace="$OUT/traces/$stem.jsonl"
  test ! -e "$trace" || { echo "REFUSE_OVERWRITE $trace"; exit 6; }
  echo "BATCH_START tier=$TIER suite=$suite task=$task trials=$trials $(date -Iseconds)"
  cd "$ROOT"
  PYTHONUTF8=1 LIBERO_CONFIG_PATH=/root/gpufree-data/libero-config \
  PYTHONPATH="$ROOT/third_party/libero" \
    examples/libero/.venv/bin/python examples/libero/monitoring_main.py \
      --args.task-suite-name "$suite" --args.task-id "$task" \
      --args.num-trials-per-task "$trials" --args.replan-steps 5 \
      --args.seed "$SEED" --args.sampling-noise-seed "$NOISE_SEED" \
      --args.disturbance-num-steps 0 --args.translation-action-scale 1.0 \
      --args.trace-out-path "$trace" --args.video-out-path "$OUT/videos" \
      --args.visual-out-path "$OUT/visual_sidecars" --args.visual-stride 1 \
      >"$OUT/logs/$stem.log" 2>&1
  echo "BATCH_DONE tier=$TIER suite=$suite task=$task $(date -Iseconds)"
}

for task in 0 2 4 6 8; do run_one libero_spatial "$task" 4; done
for task in 0 2 4 6 8; do run_one libero_object  "$task" 4; done
for task in 0 2 4 6 8; do run_one libero_goal    "$task" 4; done
for task in 0 9 19 29 39; do run_one libero_90   "$task" 4; done

printf 'tier=%s\nseed=%s\nnoise_seed=%s\ncompleted=%s\n' \
  "$TIER" "$SEED" "$NOISE_SEED" "$(date -Iseconds)" > "$OUT/COLLECTION_COMPLETE"
echo "TIER_COMPLETE $OUT"
