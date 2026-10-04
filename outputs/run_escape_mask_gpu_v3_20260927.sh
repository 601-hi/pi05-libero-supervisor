#!/usr/bin/env bash
set -euo pipefail

OPENPI=/root/gpufree-data/vla-workspace/openpi
TOOLS=/root/gpufree-data/supervisor-tools-v1
OUT=/root/gpufree-data/libero-traces/goal_relation_direct_rollback_v3_mask
mkdir -p "$OUT" "$OUT/videos"

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
HF_HOME=/root/gpufree-data/hf-cache \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
nohup /root/gpufree-data/vision-env/bin/python \
  scripts/run_articulation_vision_sidecar.py \
  --query "open top drawer" --calibrated \
  >"$OUT/vision.log" 2>&1 &
echo $! >"$OUT/vision.pid"

for _ in $(seq 1 120); do
  if grep -q 'server listening on 0.0.0.0:8000' "$OUT/policy.log" 2>/dev/null \
      && grep -q '"event": "ready"' "$OUT/vision.log" 2>/dev/null; then
    break
  fi
  sleep 2
done
grep -q 'server listening on 0.0.0.0:8000' "$OUT/policy.log"
grep -q '"event": "ready"' "$OUT/vision.log"

cd "$OPENPI/examples/libero"
LIBERO_CONFIG_PATH=/root/gpufree-data/libero-config \
PYTHONPATH="$OPENPI/third_party/libero" \
.venv/bin/python monitoring_main.py \
  --args.task-suite-name libero_90 \
  --args.task-id 0 \
  --args.num-trials-per-task 5 \
  --args.seed 7 \
  --args.sampling-noise-seed 2026092700 \
  --args.replan-steps 5 \
  --args.trace-out-path "$OUT/control_task0_seed7_noise2026092700_5ep.jsonl" \
  --args.video-out-path "$OUT/videos" \
  --args.supervisor-enabled \
  --args.supervisor-intervention-enabled \
  --args.supervisor-execution-mode frozen_four_state_articulation \
  --args.supervisor-articulation-control-enabled \
  --args.supervisor-articulation-isolated-control \
  --args.supervisor-rollback-enabled \
  --args.supervisor-checkpoint-oracle-enabled \
  --args.no-supervisor-recovery-prompt-enabled \
  --args.supervisor-rollback-step-budget 320 \
  --args.supervisor-rollback-replay-step-budget 160 \
  >"$OUT/control.log" 2>&1

echo "MASK_GPU_V3_DONE $(date --iso-8601=seconds)" | tee "$OUT/DONE"
