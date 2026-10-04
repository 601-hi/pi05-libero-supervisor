#!/usr/bin/env bash
set -euo pipefail

# GPU collection only. Start the pi0.5 policy server separately on port 8000.
# This stage intentionally contains no action disturbance.
ROOT=/root/gpufree-data/vla-workspace/openpi
OUT=/root/gpufree-data/libero-traces/natural_failure_stage_a_seed34
TRACE="$OUT/traces"
VISUAL="$OUT/visual_sidecars"
VIDEO="$OUT/videos"
LOG="$OUT/logs"
mkdir -p "$TRACE" "$VISUAL" "$VIDEO" "$LOG"
cd "$ROOT"

available_kib=$(df --output=avail /root/gpufree-data | tail -1 | tr -d ' ')
minimum_kib=$((8 * 1024 * 1024))
if (( available_kib < minimum_kib )); then
  echo "INSUFFICIENT_DISK_KIB available=$available_kib required=$minimum_kib"
  exit 5
fi
echo "DISK_PREFLIGHT_OK_KIB available=$available_kib required=$minimum_kib"

ss -ltn 'sport = :8000' | grep -q LISTEN || {
  echo "POLICY_SERVER_NOT_LISTENING_ON_8000"
  exit 3
}

run_one() {
  local suite="$1" task="$2" trials="$3"
  local stem="${suite}_task${task}_seed34_natural_${trials}ep"
  local trace="$TRACE/${stem}.jsonl"
  test ! -e "$trace" || { echo "REFUSE_OVERWRITE $trace"; exit 2; }
  echo "COLLECTION_START suite=$suite task=$task trials=$trials $(date -Iseconds)"
  PYTHONUTF8=1 LIBERO_CONFIG_PATH=/root/gpufree-data/libero-config \
  PYTHONPATH="$ROOT/third_party/libero" \
    examples/libero/.venv/bin/python examples/libero/monitoring_main.py \
      --args.task-suite-name "$suite" \
      --args.task-id "$task" \
      --args.num-trials-per-task "$trials" \
      --args.replan-steps 5 \
      --args.seed 34 \
      --args.sampling-noise-seed 2026091134 \
      --args.disturbance-num-steps 0 \
      --args.translation-action-scale 1.0 \
      --args.trace-out-path "$trace" \
      --args.video-out-path "$VIDEO" \
      --args.visual-out-path "$VISUAL" \
      --args.visual-stride 1 \
      >"$LOG/${stem}.log" 2>&1
  echo "COLLECTION_DONE suite=$suite task=$task $(date -Iseconds)"
}

# Interleave historically unstable tasks and stable controls to reduce run-order bias.
run_one libero_spatial 4 15
run_one libero_spatial 0 10
run_one libero_90 0 15
run_one libero_spatial 9 15
run_one libero_90 9 5
run_one libero_spatial 3 15
run_one libero_spatial 1 10
run_one libero_90 29 15
run_one libero_spatial 7 15
run_one libero_90 19 5

echo "NATURAL_FAILURE_STAGE_A_COMPLETE $(date -Iseconds)"
