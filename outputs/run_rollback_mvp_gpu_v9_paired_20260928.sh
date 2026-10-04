#!/usr/bin/env bash
set -euo pipefail

OPENPI=/root/gpufree-data/vla-workspace/openpi
TOOLS=/root/gpufree-data/supervisor-tools-v1
OUT=${OUT:-/root/gpufree-data/libero-traces/goal_relation_rollback_mvp_v9_paired}
TRIALS=${TRIALS:-5}
EPISODES=${EPISODES:-}
mkdir -p "$OUT" "$OUT/videos_strict" "$OUT/videos_mvp"

cleanup() {
  if [[ -f "$OUT/policy.pid" ]]; then kill "$(cat "$OUT/policy.pid")" 2>/dev/null || true; fi
  if [[ -f "$OUT/vision.pid" ]]; then kill "$(cat "$OUT/vision.pid")" 2>/dev/null || true; fi
}
trap cleanup EXIT

for path in "$OUT/strict.jsonl" "$OUT/mvp.jsonl" "$OUT/DONE"; do
  if [[ -e "$path" ]]; then
    echo "Refusing to overwrite existing result: $path" >&2
    exit 2
  fi
done

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
  --args.num-trials-per-task "$TRIALS" --args.seed 7
  --args.sampling-noise-seed 2026092700 --args.replan-steps 5
  --args.supervisor-enabled --args.supervisor-intervention-enabled
  --args.supervisor-execution-mode frozen_four_state_articulation
  --args.supervisor-articulation-control-enabled
  --args.supervisor-articulation-isolated-control
  --args.supervisor-rollback-enabled
  --args.supervisor-escape-cycle-breaker-control-enabled
  --args.no-supervisor-safe-step-pulse-control-enabled
  --args.no-supervisor-escape-progress-mask-control-enabled
  --args.supervisor-checkpoint-oracle-enabled
  --args.no-supervisor-recovery-prompt-enabled
  --args.supervisor-rollback-adaptive-budget-enabled
  --args.supervisor-rollback-step-budget 320
  --args.supervisor-rollback-replay-step-budget 160
  --args.supervisor-rollback-hard-step-budget 640
  --args.supervisor-rollback-hard-replay-step-budget 480
  --args.supervisor-rollback-budget-extension-chunk 80
)
if [[ -n "$EPISODES" ]]; then
  COMMON+=(--args.episode-indices "$EPISODES")
fi

run_one() {
  local name=$1
  shift
  LIBERO_CONFIG_PATH=/root/gpufree-data/libero-config \
  PYTHONPATH="$OPENPI/third_party/libero" \
  .venv/bin/python monitoring_main.py "${COMMON[@]}" "$@" \
    --args.trace-out-path "$OUT/$name.jsonl" \
    --args.video-out-path "$OUT/videos_$name" \
    >"$OUT/$name.log" 2>&1
}

run_one strict \
  --args.no-supervisor-rollback-mvp-near-trajectory-enabled \
  --args.no-supervisor-replan-novelty-control-enabled

run_one mvp \
  --args.supervisor-rollback-mvp-near-trajectory-enabled \
  --args.supervisor-rollback-position-tolerance-m 0.012 \
  --args.supervisor-rollback-orientation-tolerance-rad 0.12 \
  --args.supervisor-replan-novelty-control-enabled \
  --args.supervisor-replan-max-rollback-levels 3

python3 "$TOOLS/outputs/audit_rollback_mvp_gpu_v9_20260928.py" \
  --trace-dir "$OUT" --output "$OUT/audit.json"
echo "ROLLBACK_MVP_GPU_PAIRED_DONE trials=$TRIALS $(date --iso-8601=seconds)" | tee "$OUT/DONE"

