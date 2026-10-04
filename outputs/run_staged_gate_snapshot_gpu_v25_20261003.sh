#!/usr/bin/env bash
set -euo pipefail

OPENPI=/root/gpufree-data/vla-workspace/openpi
TOOLS=/root/gpufree-data/supervisor-tools-v1
MODE=${MODE:-capture}
OUT=${OUT:-/root/gpufree-data/libero-traces/staged_gate_snapshot_v25_${MODE}_ep2}
SNAPSHOT=${SNAPSHOT:-}
REPLAY_ESCAPE_MAX_STEPS=${REPLAY_ESCAPE_MAX_STEPS:-6}
REPLAY_ESCAPE_STEP_BUDGET=${REPLAY_ESCAPE_STEP_BUDGET:-80}
MINIMUM_SPATIAL_RETREAT_M=${MINIMUM_SPATIAL_RETREAT_M:-0.10}

if [[ "$MODE" != "capture" && "$MODE" != "load" ]]; then
  echo "MODE must be capture or load" >&2
  exit 2
fi
if [[ "$MODE" == "load" && -z "$SNAPSHOT" ]]; then
  echo "SNAPSHOT is required in load mode" >&2
  exit 2
fi

mkdir -p "$OUT" "$OUT/videos_mvp" "$OUT/snapshots"
cleanup() {
  [[ -f "$OUT/policy.pid" ]] && kill "$(cat "$OUT/policy.pid")" 2>/dev/null || true
  [[ -f "$OUT/vision.pid" ]] && kill "$(cat "$OUT/vision.pid")" 2>/dev/null || true
}
trap cleanup EXIT
for path in "$OUT/mvp.jsonl" "$OUT/DONE"; do
  [[ ! -e "$path" ]] || { echo "Refusing to overwrite $path" >&2; exit 2; }
done

cd "$OPENPI"
nohup .venv/bin/python scripts/serve_policy.py policy:checkpoint \
  --policy.config pi05_libero \
  --policy.dir /root/gpufree-data/openpi-data/openpi-assets/checkpoints/pi05_libero \
  >"$OUT/policy.log" 2>&1 &
echo $! >"$OUT/policy.pid"
cd "$TOOLS"
HF_HOME=/root/gpufree-data/hf-cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
nohup /root/gpufree-data/vision-env/bin/python scripts/run_articulation_vision_sidecar.py \
  --query "open top drawer" --calibrated >"$OUT/vision.log" 2>&1 &
echo $! >"$OUT/vision.pid"
for _ in $(seq 1 120); do
  if grep -q 'server listening on 0.0.0.0:8000' "$OUT/policy.log" 2>/dev/null \
      && grep -q '"event": "ready"' "$OUT/vision.log" 2>/dev/null; then break; fi
  sleep 2
done
grep -q 'server listening on 0.0.0.0:8000' "$OUT/policy.log"
grep -q '"event": "ready"' "$OUT/vision.log"

SNAPSHOT_ARGS=()
if [[ "$MODE" == "capture" ]]; then
  SNAPSHOT_ARGS=(--args.supervisor-fault-snapshot-path "$OUT/snapshots/fault_task{task_id:02d}_ep{episode_idx:03d}_a{action_index:04d}.npz")
else
  SNAPSHOT_ARGS=(--args.supervisor-fault-snapshot-load-path "$SNAPSHOT")
fi

cd "$OPENPI/examples/libero"
LIBERO_CONFIG_PATH=/root/gpufree-data/libero-config PYTHONPATH="$OPENPI/third_party/libero:$TOOLS" \
.venv/bin/python monitoring_main.py \
  --args.task-suite-name libero_90 --args.task-id 0 \
  --args.num-trials-per-task 5 --args.episode-indices 2 \
  --args.seed 7 --args.sampling-noise-seed 2026092700 --args.replan-steps 5 \
  --args.supervisor-enabled --args.supervisor-intervention-enabled \
  --args.supervisor-execution-mode frozen_four_state_articulation \
  --args.supervisor-articulation-control-enabled \
  --args.supervisor-articulation-isolated-control \
  --args.supervisor-rollback-enabled \
  --args.supervisor-escape-cycle-breaker-control-enabled \
  --args.no-supervisor-safe-step-pulse-control-enabled \
  --args.no-supervisor-escape-progress-mask-control-enabled \
  --args.supervisor-checkpoint-oracle-enabled \
  --args.no-supervisor-recovery-prompt-enabled \
  --args.supervisor-rollback-adaptive-budget-enabled \
  --args.supervisor-rollback-step-budget 400 \
  --args.supervisor-rollback-replay-step-budget 320 \
  --args.supervisor-rollback-hard-step-budget 800 \
  --args.supervisor-rollback-hard-replay-step-budget 640 \
  --args.supervisor-rollback-budget-extension-chunk 80 \
  --args.supervisor-rollback-mvp-near-trajectory-enabled \
  --args.supervisor-rollback-position-tolerance-m 0.006 \
  --args.supervisor-rollback-orientation-tolerance-rad 0.08 \
  --args.supervisor-direct-joint-replay-enabled \
  --args.supervisor-joint-replay-maximum-step-rad 0.02 \
  --args.supervisor-joint-replay-maximum-step-change-rad 0.006 \
  --args.supervisor-joint-replay-tolerance-rad 0.025 \
  --args.supervisor-replan-novelty-control-enabled \
  --args.supervisor-replan-max-rollback-levels 3 \
  --args.supervisor-rollback-minimum-replan-steps 20 \
  --args.supervisor-rollback-replan-step-increment 15 \
  --args.supervisor-rollback-minimum-history-depth 18 \
  --args.supervisor-rollback-history-depth-increment 12 \
  --args.supervisor-rollback-minimum-spatial-retreat-m "$MINIMUM_SPATIAL_RETREAT_M" \
  --args.supervisor-rollback-spatial-retreat-increment-m 0.05 \
  --args.supervisor-rollback-maximum-spatial-retreat-m 0.18 \
  --args.supervisor-replay-temporary-escape-max-steps "$REPLAY_ESCAPE_MAX_STEPS" \
  --args.supervisor-replay-temporary-escape-step-budget "$REPLAY_ESCAPE_STEP_BUDGET" \
  "${SNAPSHOT_ARGS[@]}" \
  --args.trace-out-path "$OUT/mvp.jsonl" \
  --args.video-out-path "$OUT/videos_mvp" >"$OUT/mvp.log" 2>&1
echo "STAGED_GATE_SNAPSHOT_V25_${MODE}_DONE $(date --iso-8601=seconds)" | tee "$OUT/DONE"
