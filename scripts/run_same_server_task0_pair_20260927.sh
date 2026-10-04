#!/usr/bin/env bash
set -euo pipefail

OPENPI=/root/gpufree-data/vla-workspace/openpi
TOOLS=/root/gpufree-data/supervisor-tools-v1
OUT=/root/gpufree-data/libero-traces/natural_failure_same_server_pair_20260927
BASE="$OUT/libero90_t0_seed34_baseline.jsonl"
CONTROL="$OUT/libero90_t0_seed34_control.jsonl"

if [[ -e "$BASE" || -e "$CONTROL" ]]; then
  echo "Refusing to overwrite frozen pair output: $OUT" >&2
  exit 2
fi
mkdir -p "$OUT/videos/baseline" "$OUT/videos/control"
cd "$OPENPI"
export LIBERO_CONFIG_PATH=/root/gpufree-data/libero-config
export PYTHONPATH="$OPENPI/third_party/libero:$TOOLS"

COMMON=(
  --args.task-suite-name libero_90
  --args.task-id 0
  --args.num-trials-per-task 1
  --args.replan-steps 5
  --args.seed 34
  --args.sampling-noise-seed 2026091134
)

echo "PAIR_BASELINE_START $(date --iso-8601=seconds)"
examples/libero/.venv/bin/python examples/libero/monitoring_main.py \
  "${COMMON[@]}" \
  --args.trace-out-path "$BASE" \
  --args.video-out-path "$OUT/videos/baseline"
echo "PAIR_BASELINE_DONE $(date --iso-8601=seconds)"

echo "PAIR_CONTROL_START $(date --iso-8601=seconds)"
examples/libero/.venv/bin/python examples/libero/monitoring_main.py \
  "${COMMON[@]}" \
  --args.trace-out-path "$CONTROL" \
  --args.video-out-path "$OUT/videos/control" \
  --args.supervisor-enabled \
  --args.supervisor-intervention-enabled \
  --args.supervisor-execution-mode frozen_four_state \
  --args.supervisor-checkpoint-audit-enabled \
  --args.supervisor-checkpoint-oracle-enabled \
  --args.supervisor-rollback-enabled \
  --args.supervisor-rollback-step-budget 320 \
  --args.supervisor-rollback-replay-step-budget 160 \
  --args.supervisor-checkpoint-minimum-reliability 0.2588455491502583 \
  --args.supervisor-checkpoint-contact-zero-band-m 1e-05
echo "PAIR_CONTROL_DONE $(date --iso-8601=seconds)"

cd "$TOOLS"
"$OPENPI/.venv/bin/python" scripts/audit_same_server_task0_pair_20260927.py
