#!/usr/bin/env bash
set -euo pipefail
cd /root/gpufree-data/supervisor-tools-v1
PY=/root/gpufree-data/vla-workspace/openpi/examples/libero/.venv/bin/python
R=/root/gpufree-data/supervisor-results/visual_latent_v2
mkdir -p "$R/candidates"

run_one() {
  local name="$1" kind="$2" margin="$3"
  "$PY" visual_latent_v2/train_semantic_visual_head.py \
    --dataset "$R/train_semantic_chunks.npz" --features "$R/train_head_features.npz" \
    --kind "$kind" --margin "$margin" --margin-weight 0.25 --epochs 300 --seed 20260904 \
    --out "$R/candidates/${name}.pt" --report "$R/candidates/${name}_train.json"
  for budget in 0 1; do
    "$PY" visual_latent_v2/calibrate_semantic_visual_head.py \
      --dataset "$R/calibration_semantic_chunks.npz" --features "$R/calibration_head_features.npz" \
      --model "$R/candidates/${name}.pt" \
      --base-evaluation "$R/calibration_frozen_fusion.json" \
      --max-normal-episode-fp "$budget" --out "$R/candidates/${name}_calibration_budget${budget}.json"
  done
}

run_one logistic_m0 logistic 0.0
run_one logistic_m05 logistic 0.5
run_one logistic_m1 logistic 1.0
run_one mlp_m05 mlp_2layer 0.5
run_one mlp_m1 mlp_2layer 1.0
sha256sum "$R"/candidates/* > "$R/candidates/SHA256SUMS.txt"
echo ALL_SEMANTIC_HEAD_CANDIDATES_DONE
