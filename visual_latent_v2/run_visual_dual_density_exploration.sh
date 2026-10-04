#!/usr/bin/env bash
set -euo pipefail
cd /root/gpufree-data/supervisor-tools-v1
PY=/root/gpufree-data/vla-workspace/openpi/examples/libero/.venv/bin/python
R=/root/gpufree-data/supervisor-results/visual_latent_v2
"$PY" -m py_compile visual_latent_v2/visual_dual_density.py visual_latent_v2/fit_visual_dual_density.py visual_latent_v2/evaluate_visual_dual_density.py
"$PY" visual_latent_v2/fit_visual_dual_density.py \
 --train-data "$R/train_semantic_chunks.npz" --train-features "$R/train_head_features.npz" \
 --cal-data "$R/calibration_semantic_chunks.npz" --cal-features "$R/calibration_head_features.npz" \
 --base "$R/calibration_frozen_fusion.json" --budget 1 \
 --out-model "$R/visual_dual_density_v2_exploratory.npz" --out-config "$R/visual_dual_density_v2_exploratory.json"
"$PY" visual_latent_v2/evaluate_visual_dual_density.py \
 --data "$R/test_seed21_semantic_chunks.npz" --features "$R/test_seed21_head_features.npz" \
 --model "$R/visual_dual_density_v2_exploratory.npz" --config "$R/visual_dual_density_v2_exploratory.json" \
 --base "$R/test_seed21_base_fusion.json" --out "$R/visual_dual_density_v2_seed21_diagnostic.json"
echo VISUAL_DUAL_DENSITY_EXPLORATION_DONE
