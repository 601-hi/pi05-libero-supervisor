#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/gpufree-data/vla-workspace/openpi
OUT=/root/gpufree-data/libero-traces/visual_latent_v2/test_seed21
PY="$ROOT/examples/libero/.venv/bin/python"
mkdir -p "$OUT/traces" "$OUT/latents" "$OUT/videos" "$OUT/logs"

run_one() {
  local condition="$1" scale="$2" steps="$3"
  local tag="test_seed21_${condition}"
  local trace="$OUT/traces/${tag}.jsonl"
  local latent="$OUT/latents/${tag}"
  if [[ -e "$trace" || -e "$latent" ]]; then echo "REFUSING_TO_OVERWRITE $tag" >&2; exit 2; fi
  mkdir -p "$latent" "$OUT/videos/$tag"
  echo "TEST_BATCH_START tag=$tag $(date --iso-8601=seconds)" | tee -a "$OUT/orchestrator.log"
  cd "$ROOT"
  LIBERO_CONFIG_PATH=/root/gpufree-data/libero-config PYTHONPATH="$ROOT/third_party/libero" MUJOCO_GL=egl \
    "$PY" examples/libero/monitoring_main.py \
      --args.task-suite-name libero_spatial --args.num-trials-per-task 2 \
      --args.replan-steps 5 --args.seed 21 --args.sampling-noise-seed 2026090321 \
      --args.disturbance-start-mode random --args.disturbance-random-start-min 20 \
      --args.disturbance-random-start-max 90 --args.disturbance-num-steps "$steps" \
      --args.translation-action-scale "$scale" --args.trace-out-path "$trace" \
      --args.video-out-path "$OUT/videos/$tag" --args.visual-latent-out-path "$latent" \
      > "$OUT/logs/$tag.log" 2>&1
  local ends sidecars
  ends=$(grep -c '"event": "episode_end"' "$trace")
  sidecars=$(find "$latent" -maxdepth 1 -type f -name '*.npz' | wc -l)
  [[ "$ends" -eq 20 && "$sidecars" -eq 20 ]]
  echo "TEST_BATCH_DONE tag=$tag episode_ends=$ends sidecars=$sidecars $(date --iso-8601=seconds)" | tee -a "$OUT/orchestrator.log"
}
run_one normal 1.0 0
run_one scale025 0.25 10
run_one scale050 0.5 10
run_one scale075 0.75 10
echo "ALL_FROZEN_TEST_BATCHES_DONE $(date --iso-8601=seconds)" | tee -a "$OUT/orchestrator.log"
