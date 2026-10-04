#!/usr/bin/env bash
set -euo pipefail

cd /root/gpufree-data/supervisor-tools-v1
export HF_HOME=/root/gpufree-data/hf-cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

for episode in 0 1 2 4; do
  sidecar="/root/gpufree-data/libero-traces/natural_validation_seed35_v1/visual_sidecars/rollout_libero_90_seed35_noise2026091835_task00_episode00${episode}_fixed_startnone_n000_scale1p000_failure.npz"
  output="outputs/task0_close_relation_20260927/holdout_failure35_ep${episode}_counterstate.json"
  /root/gpufree-data/vision-env/bin/python scripts/score_articulation_counterstate.py \
    --sidecar "${sidecar}" \
    --output "${output}" \
    --query "open top drawer" \
    --stride 4
done
