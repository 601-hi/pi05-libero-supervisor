#!/usr/bin/env bash
set -euo pipefail
TOOLS=/root/gpufree-data/supervisor-tools-v1/visual_latent_v2
PY=/root/gpufree-data/vla-workspace/openpi/.venv/bin/python
NEW=/root/gpufree-data/libero-traces/visual_latent_v2/train_calibration
OLD=/root/gpufree-data/libero-traces/abnormal_expert_v1

for tag in \
  train_seed18_normal train_seed18_scale025 train_seed18_scale050 train_seed18_scale075 \
  train_seed19_normal train_seed19_scale025 train_seed19_scale050 train_seed19_scale075 \
  calibration_seed20_normal calibration_seed20_scale025 calibration_seed20_scale050 calibration_seed20_scale075
do
  echo "VALIDATE_START $tag"
  "$PY" "$TOOLS/verify_visual_latent_artifacts.py" \
    --trace "$NEW/traces/$tag.jsonl" --latent-dir "$NEW/latents/$tag"
  "$PY" "$TOOLS/compare_action_invariance.py" \
    --off "$OLD/$tag.jsonl" --on "$NEW/traces/$tag.jsonl"
  echo "VALIDATE_DONE $tag"
done
echo "ALL_COLLECTION_VALIDATION_DONE"
du -sh "$NEW"
