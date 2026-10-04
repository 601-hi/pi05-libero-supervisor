#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/gpufree-data/vla-workspace/openpi
OUT=/root/gpufree-data/libero-traces/visual_latent_v2/train_calibration
LOG="$OUT/logs"
PY="$ROOT/examples/libero/.venv/bin/python"
mkdir -p "$OUT/traces" "$OUT/latents" "$OUT/videos" "$LOG"

run_one() {
  local split="$1" seed="$2" condition="$3" scale="$4" steps="$5"
  local tag="${split}_seed${seed}_${condition}"
  local trace="$OUT/traces/${tag}.jsonl"
  local latent="$OUT/latents/${tag}"
  local video="$OUT/videos/${tag}"
  local noise_seed=$((2026090300 + seed))
  if [[ -e "$trace" || -e "$latent" ]]; then
    echo "REFUSING_TO_OVERWRITE tag=$tag" >&2
    exit 2
  fi
  mkdir -p "$latent" "$video"
  echo "BATCH_START tag=$tag $(date --iso-8601=seconds)" | tee -a "$OUT/orchestrator.log"
  cd "$ROOT"
  LIBERO_CONFIG_PATH=/root/gpufree-data/libero-config \
  PYTHONPATH="$ROOT/third_party/libero" MUJOCO_GL=egl \
    "$PY" examples/libero/monitoring_main.py \
      --args.task-suite-name libero_spatial \
      --args.num-trials-per-task 1 \
      --args.replan-steps 5 \
      --args.seed "$seed" \
      --args.sampling-noise-seed "$noise_seed" \
      --args.disturbance-start-mode random \
      --args.disturbance-random-start-min 20 \
      --args.disturbance-random-start-max 90 \
      --args.disturbance-num-steps "$steps" \
      --args.translation-action-scale "$scale" \
      --args.trace-out-path "$trace" \
      --args.video-out-path "$video" \
      --args.visual-latent-out-path "$latent" \
      > "$LOG/${tag}.log" 2>&1
  local ends sidecars
  ends=$(grep -c '"event": "episode_end"' "$trace")
  sidecars=$(find "$latent" -maxdepth 1 -type f -name '*.npz' | wc -l)
  if [[ "$ends" -ne 10 || "$sidecars" -ne 10 ]]; then
    echo "BATCH_INCOMPLETE tag=$tag episode_ends=$ends sidecars=$sidecars" >&2
    exit 3
  fi
  echo "BATCH_DONE tag=$tag episode_ends=$ends sidecars=$sidecars $(date --iso-8601=seconds)" | tee -a "$OUT/orchestrator.log"
}

for seed in 18 19; do
  for spec in "normal 1.0 0" "scale025 0.25 10" "scale050 0.5 10" "scale075 0.75 10"; do
    read -r condition scale steps <<< "$spec"
    run_one train "$seed" "$condition" "$scale" "$steps"
  done
done
for spec in "normal 1.0 0" "scale025 0.25 10" "scale050 0.5 10" "scale075 0.75 10"; do
  read -r condition scale steps <<< "$spec"
  run_one calibration 20 "$condition" "$scale" "$steps"
done
echo "ALL_TRAIN_CALIBRATION_BATCHES_DONE $(date --iso-8601=seconds)" | tee -a "$OUT/orchestrator.log"
