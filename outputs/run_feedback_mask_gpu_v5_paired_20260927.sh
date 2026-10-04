#!/usr/bin/env bash
set -euo pipefail

OPENPI=/root/gpufree-data/vla-workspace/openpi
TOOLS=/root/gpufree-data/supervisor-tools-v1
OUT=/root/gpufree-data/libero-traces/goal_relation_direct_rollback_v5_feedback_mask_paired
mkdir -p "$OUT" "$OUT/videos_baseline" "$OUT/videos_control"

cleanup() {
  if [[ -f "$OUT/policy.pid" ]]; then kill "$(cat "$OUT/policy.pid")" 2>/dev/null || true; fi
  if [[ -f "$OUT/vision.pid" ]]; then kill "$(cat "$OUT/vision.pid")" 2>/dev/null || true; fi
}
trap cleanup EXIT

cd "$OPENPI"
nohup .venv/bin/python scripts/serve_policy.py policy:checkpoint \
  --policy.config pi05_libero \
  --policy.dir /root/gpufree-data/openpi-data/openpi-assets/checkpoints/pi05_libero \
  >"$OUT/policy.log" 2>&1 &
echo $! >"$OUT/policy.pid"

cd "$TOOLS"
HF_HOME=/root/gpufree-data/hf-cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
nohup /root/gpufree-data/vision-env/bin/python \
  scripts/run_articulation_vision_sidecar.py \
  --query "open top drawer" --calibrated \
  >"$OUT/vision.log" 2>&1 &
echo $! >"$OUT/vision.pid"

for _ in $(seq 1 120); do
  if grep -q 'server listening on 0.0.0.0:8000' "$OUT/policy.log" 2>/dev/null \
      && grep -q '"event": "ready"' "$OUT/vision.log" 2>/dev/null; then break; fi
  sleep 2
done
grep -q 'server listening on 0.0.0.0:8000' "$OUT/policy.log"
grep -q '"event": "ready"' "$OUT/vision.log"

cd "$OPENPI/examples/libero"
COMMON=(
  --args.task-suite-name libero_90 --args.task-id 0
  --args.num-trials-per-task 5 --args.seed 7
  --args.sampling-noise-seed 2026092700 --args.replan-steps 5
)
LIBERO_CONFIG_PATH=/root/gpufree-data/libero-config \
PYTHONPATH="$OPENPI/third_party/libero" \
.venv/bin/python monitoring_main.py "${COMMON[@]}" \
  --args.trace-out-path "$OUT/baseline.jsonl" \
  --args.video-out-path "$OUT/videos_baseline" \
  >"$OUT/baseline.log" 2>&1

LIBERO_CONFIG_PATH=/root/gpufree-data/libero-config \
PYTHONPATH="$OPENPI/third_party/libero" \
.venv/bin/python monitoring_main.py "${COMMON[@]}" \
  --args.trace-out-path "$OUT/control.jsonl" \
  --args.video-out-path "$OUT/videos_control" \
  --args.supervisor-enabled \
  --args.supervisor-intervention-enabled \
  --args.supervisor-execution-mode frozen_four_state_articulation \
  --args.supervisor-articulation-control-enabled \
  --args.supervisor-articulation-isolated-control \
  --args.supervisor-rollback-enabled \
  --args.supervisor-escape-progress-mask-control-enabled \
  --args.supervisor-checkpoint-oracle-enabled \
  --args.no-supervisor-recovery-prompt-enabled \
  --args.supervisor-rollback-step-budget 320 \
  --args.supervisor-rollback-replay-step-budget 160 \
  >"$OUT/control.log" 2>&1

echo "FEEDBACK_MASK_GPU_V5_PAIRED_DONE $(date --iso-8601=seconds)" | tee "$OUT/DONE"
