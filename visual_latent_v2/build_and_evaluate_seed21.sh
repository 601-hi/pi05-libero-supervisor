#!/usr/bin/env bash
set -euo pipefail
cd /root/gpufree-data/supervisor-tools-v1
PY=/root/gpufree-data/vla-workspace/openpi/examples/libero/.venv/bin/python
T=/root/gpufree-data/libero-traces/visual_latent_v2/test_seed21
R=/root/gpufree-data/supervisor-results/visual_latent_v2

for tag in test_seed21_normal test_seed21_scale025 test_seed21_scale050 test_seed21_scale075; do
  "$PY" visual_latent_v2/verify_visual_latent_artifacts.py --trace "$T/traces/$tag.jsonl" --latent-dir "$T/latents/$tag"
done
"$PY" export_dynamics_expert_dataset.py --require-random-or-event \
  --trace "$T/traces/test_seed21_normal.jsonl" --trace "$T/traces/test_seed21_scale025.jsonl" \
  --trace "$T/traces/test_seed21_scale050.jsonl" --trace "$T/traces/test_seed21_scale075.jsonl" \
  --out "$R/test_seed21_dataset.npz"
"$PY" score_dynamics_experts.py --dataset "$R/test_seed21_dataset.npz" \
  --normal /root/gpufree-data/supervisor-results/conditional_normal_dynamics_expert_frozen_v2.pt \
  --abnormal-expert /root/gpufree-data/supervisor-results/abnormal_dynamics_expert_frozen_v1.pt \
  --out "$R/test_seed21_scores.npz"
"$PY" evaluate_two_expert_fusion.py --scores "$R/test_seed21_scores.npz" \
  --config /root/gpufree-data/supervisor-results/two_expert_fusion_budget1_v3.json \
  --out "$R/test_seed21_base_fusion.json"
"$PY" visual_latent_v2/export_semantic_chunk_dataset.py \
  --dataset "$R/test_seed21_dataset.npz" --scores "$R/test_seed21_scores.npz" \
  --fusion /root/gpufree-data/supervisor-results/two_expert_fusion_budget1_v3.json \
  --latent-source "0=$T/latents/test_seed21_normal" --latent-source "1=$T/latents/test_seed21_scale025" \
  --latent-source "2=$T/latents/test_seed21_scale050" --latent-source "3=$T/latents/test_seed21_scale075" \
  --out "$R/test_seed21_semantic_chunks.npz"
"$PY" visual_latent_v2/cache_semantic_head_features.py --dataset "$R/test_seed21_semantic_chunks.npz" \
  --out "$R/test_seed21_head_features.npz" --projection-dim 64 --seed 20260904
"$PY" visual_latent_v2/evaluate_frozen_semantic_cascade.py \
  --dataset "$R/test_seed21_semantic_chunks.npz" --features "$R/test_seed21_head_features.npz" \
  --model "$R/candidates/mlp_m1.pt" --base-evaluation "$R/test_seed21_base_fusion.json" \
  --frozen-manifest visual_latent_v2/FROZEN_VISUAL_CASCADE_V1.json \
  --out "$R/test_seed21_frozen_visual_cascade.json"
sha256sum "$R"/test_seed21* visual_latent_v2/FROZEN_VISUAL_CASCADE_V1.json > "$R/TEST_SEED21_SHA256SUMS.txt"
echo FROZEN_SEED21_EVALUATION_DONE
