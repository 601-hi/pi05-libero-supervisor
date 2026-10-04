#!/usr/bin/env bash
set -euo pipefail
cd /root/gpufree-data/supervisor-tools-v1
PY=/root/gpufree-data/vla-workspace/openpi/examples/libero/.venv/bin/python
N=/root/gpufree-data/libero-traces/visual_latent_v2/train_calibration/traces
L=/root/gpufree-data/libero-traces/visual_latent_v2/train_calibration/latents
R=/root/gpufree-data/supervisor-results/visual_latent_v2
mkdir -p "$R"

"$PY" export_dynamics_expert_dataset.py --require-random-or-event \
  --trace "$N/train_seed18_normal.jsonl" --trace "$N/train_seed18_scale025.jsonl" \
  --trace "$N/train_seed18_scale050.jsonl" --trace "$N/train_seed18_scale075.jsonl" \
  --trace "$N/train_seed19_normal.jsonl" --trace "$N/train_seed19_scale025.jsonl" \
  --trace "$N/train_seed19_scale050.jsonl" --trace "$N/train_seed19_scale075.jsonl" \
  --out "$R/train_dataset.npz"
"$PY" export_dynamics_expert_dataset.py --require-random-or-event \
  --trace "$N/calibration_seed20_normal.jsonl" --trace "$N/calibration_seed20_scale025.jsonl" \
  --trace "$N/calibration_seed20_scale050.jsonl" --trace "$N/calibration_seed20_scale075.jsonl" \
  --out "$R/calibration_dataset.npz"

for split in train calibration; do
  "$PY" score_dynamics_experts.py \
    --dataset "$R/${split}_dataset.npz" \
    --normal /root/gpufree-data/supervisor-results/conditional_normal_dynamics_expert_frozen_v2.pt \
    --abnormal-expert /root/gpufree-data/supervisor-results/abnormal_dynamics_expert_frozen_v1.pt \
    --out "$R/${split}_scores.npz"
done
"$PY" evaluate_two_expert_fusion.py \
  --scores "$R/calibration_scores.npz" \
  --config /root/gpufree-data/supervisor-results/two_expert_fusion_budget1_v3.json \
  --out "$R/calibration_frozen_fusion.json"

"$PY" visual_latent_v2/export_semantic_chunk_dataset.py \
  --dataset "$R/train_dataset.npz" --scores "$R/train_scores.npz" \
  --fusion /root/gpufree-data/supervisor-results/two_expert_fusion_budget1_v3.json \
  --latent-source "0=$L/train_seed18_normal" --latent-source "1=$L/train_seed18_scale025" \
  --latent-source "2=$L/train_seed18_scale050" --latent-source "3=$L/train_seed18_scale075" \
  --latent-source "4=$L/train_seed19_normal" --latent-source "5=$L/train_seed19_scale025" \
  --latent-source "6=$L/train_seed19_scale050" --latent-source "7=$L/train_seed19_scale075" \
  --out "$R/train_semantic_chunks.npz"
"$PY" visual_latent_v2/export_semantic_chunk_dataset.py \
  --dataset "$R/calibration_dataset.npz" --scores "$R/calibration_scores.npz" \
  --fusion /root/gpufree-data/supervisor-results/two_expert_fusion_budget1_v3.json \
  --latent-source "0=$L/calibration_seed20_normal" --latent-source "1=$L/calibration_seed20_scale025" \
  --latent-source "2=$L/calibration_seed20_scale050" --latent-source "3=$L/calibration_seed20_scale075" \
  --out "$R/calibration_semantic_chunks.npz"
sha256sum "$R"/*.npz "$R"/*.json > "$R/SHA256SUMS.txt"
echo BUILD_CURRENT_TRAIN_CALIBRATION_DONE
