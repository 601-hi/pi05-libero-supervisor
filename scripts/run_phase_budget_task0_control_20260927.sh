#!/usr/bin/env bash
set -euo pipefail

OPENPI=/root/gpufree-data/vla-workspace/openpi
OUT=/root/gpufree-data/libero-traces/natural_failure_recovery_phase_budget_20260927
TOOLS=/root/gpufree-data/supervisor-tools-v1

mkdir -p "$OUT/videos/control"
cd "$OPENPI"
export LIBERO_CONFIG_PATH=/root/gpufree-data/libero-config
export PYTHONPATH="$OPENPI/third_party/libero:$TOOLS"

examples/libero/.venv/bin/python examples/libero/monitoring_main.py \
  --args.task-suite-name libero_90 \
  --args.task-id 0 \
  --args.num-trials-per-task 1 \
  --args.replan-steps 5 \
  --args.seed 34 \
  --args.sampling-noise-seed 2026091134 \
  --args.trace-out-path "$OUT/libero90_t0_seed34_natural_failure_dev_control.jsonl" \
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
