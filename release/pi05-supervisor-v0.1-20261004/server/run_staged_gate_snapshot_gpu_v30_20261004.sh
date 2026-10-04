#!/usr/bin/env bash
set -euo pipefail

OPENPI=/root/gpufree-data/vla-workspace/openpi
TOOLS=/root/gpufree-data/supervisor-tools-v1
MODE=${MODE:-load}
OUT=${OUT:-/root/gpufree-data/libero-traces/staged_gate_snapshot_v30_stateful_detour_${MODE}_ep2}
SNAPSHOT=${SNAPSHOT:-}
EXPECTED_SNAPSHOT_SHA256=${EXPECTED_SNAPSHOT_SHA256:-00b42df56ec082b9ca2d4b0a6cd594de92589345046ec170b1795a1a1a0da760}
REPLAY_ESCAPE_MAX_STEPS=${REPLAY_ESCAPE_MAX_STEPS:-6}
REPLAY_ESCAPE_STEP_BUDGET=${REPLAY_ESCAPE_STEP_BUDGET:-80}
MINIMUM_SPATIAL_RETREAT_M=${MINIMUM_SPATIAL_RETREAT_M:-0.095}
PROGRESS_ONLY_REPLAN=${PROGRESS_ONLY_REPLAN:-0}
PROGRESS_REPLAN_MINIMUM_SPATIAL_RETREAT_M=${PROGRESS_REPLAN_MINIMUM_SPATIAL_RETREAT_M:-0.095}
SECOND_RECOVERY_SNAPSHOT=${SECOND_RECOVERY_SNAPSHOT:-}

if [[ "$MODE" != "capture" && "$MODE" != "load" ]]; then
  echo "MODE must be capture or load" >&2
  exit 2
fi
if [[ "$MODE" == "load" && -z "$SNAPSHOT" ]]; then
  echo "SNAPSHOT is required; v30 must not silently start an unpaired capture run" >&2
  exit 2
fi
if [[ "$MODE" == "load" ]]; then
  [[ -f "$SNAPSHOT" ]] || { echo "Snapshot not found: $SNAPSHOT" >&2; exit 2; }
  ACTUAL_SNAPSHOT_SHA256=$(sha256sum "$SNAPSHOT" | awk '{print $1}')
  if [[ "$ACTUAL_SNAPSHOT_SHA256" != "$EXPECTED_SNAPSHOT_SHA256" ]]; then
    echo "Snapshot hash mismatch: $ACTUAL_SNAPSHOT_SHA256" >&2
    exit 2
  fi
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

PROGRESS_REPLAN_ARGS=()
if [[ "$PROGRESS_ONLY_REPLAN" == "1" ]]; then
  PROGRESS_REPLAN_ARGS=(
    --args.supervisor-rollback-progress-only-replan-enabled
    --args.supervisor-rollback-progress-only-replan-minimum-recovery-depth 2
    --args.supervisor-rollback-progress-only-replan-minimum-steps 20
    --args.supervisor-rollback-progress-only-replan-minimum-spatial-retreat-m "$PROGRESS_REPLAN_MINIMUM_SPATIAL_RETREAT_M"
  )
fi

SECOND_RECOVERY_SNAPSHOT_ARGS=()
if [[ -n "$SECOND_RECOVERY_SNAPSHOT" ]]; then
  SECOND_RECOVERY_SNAPSHOT_ARGS=(
    --args.supervisor-second-recovery-snapshot-path "$SECOND_RECOVERY_SNAPSHOT"
  )
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
  --args.supervisor-rollback-step-budget 1000 \
  --args.supervisor-rollback-replay-step-budget 320 \
  --args.supervisor-rollback-hard-step-budget 1200 \
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
  --args.supervisor-taskspace-upward-escape-control-enabled \
  --args.supervisor-taskspace-upward-escape-minimum-recovery-depth 2 \
  --args.supervisor-taskspace-upward-escape-step-budget 640 \
  --args.supervisor-taskspace-upward-escape-minimum-predicted-dz-m 0.0002 \
  --args.supervisor-taskspace-upward-escape-minimum-axis-alignment 0.20 \
  --args.supervisor-taskspace-upward-escape-target-actual-dz-m 0.008 \
  --args.supervisor-taskspace-upward-escape-history-height-margin-m 0.005 \
  --args.supervisor-taskspace-upward-escape-maximum-target-regression-m 0.001 \
  --args.supervisor-taskspace-escape-protected-progress-axis 1 \
  --args.supervisor-taskspace-escape-protected-progress-desired-sign -1 \
  --args.supervisor-taskspace-escape-protected-progress-regression-weight 1.25 \
  --args.supervisor-taskspace-escape-progress-protection-decay-start 0.55 \
  --args.supervisor-taskspace-escape-progress-protection-release-completion 0.85 \
  --args.supervisor-taskspace-escape-progress-protection-reactivate-completion 0.70 \
  --args.supervisor-taskspace-escape-detour-control-enabled \
  --args.supervisor-taskspace-escape-detour-stagnation-window 24 \
  --args.supervisor-taskspace-escape-detour-minimum-window-gain-m 0.0005 \
  --args.supervisor-taskspace-escape-detour-steps 8 \
  --args.supervisor-taskspace-escape-detour-probe-steps 12 \
  --args.supervisor-taskspace-escape-detour-minimum-probe-gain-m 0.0003 \
  --args.supervisor-taskspace-escape-detour-direction-cooldown-steps 48 \
  "${PROGRESS_REPLAN_ARGS[@]}" \
  "${SECOND_RECOVERY_SNAPSHOT_ARGS[@]}" \
  "${SNAPSHOT_ARGS[@]}" \
  --args.trace-out-path "$OUT/mvp.jsonl" \
  --args.video-out-path "$OUT/videos_mvp" >"$OUT/mvp.log" 2>&1
echo "STAGED_GATE_SNAPSHOT_V30_STATEFUL_DETOUR_${MODE}_DONE $(date --iso-8601=seconds)" | tee "$OUT/DONE"
