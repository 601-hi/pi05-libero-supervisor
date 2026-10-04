#!/usr/bin/env bash
set -euo pipefail
tools=/root/gpufree-data/supervisor-tools-v1
results=/root/gpufree-data/supervisor-results
py=/root/gpufree-data/vla-workspace/openpi/examples/libero/.venv/bin/python
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
cd "$tools"
run_candidate() {
  local name="$1" margin="$2" la="$3" ln="$4"
  "$py" finetune_abnormal_expert_margin.py \
    --dataset "$results/abnormal_expert_train_v1.npz" \
    --normal "$results/conditional_normal_dynamics_expert_frozen_v2.pt" \
    --initial-abnormal "$results/abnormal_dynamics_expert_frozen_v1.pt" \
    --margin-abnormal "$margin" --margin-normal 0 \
    --lambda-abnormal "$la" --lambda-normal "$ln" --epochs 150 \
    --out "$results/abnormal_dynamics_expert_margin_${name}.pt" \
    --report "$results/abnormal_dynamics_expert_margin_${name}.json"
  "$py" score_dynamics_experts.py \
    --dataset "$results/abnormal_expert_calibration_v1.npz" \
    --normal "$results/conditional_normal_dynamics_expert_frozen_v2.pt" \
    --abnormal-expert "$results/abnormal_dynamics_expert_margin_${name}.pt" \
    --out "$results/two_expert_calibration_scores_margin_${name}.npz"
  "$py" calibrate_two_expert_episode_risk.py \
    --scores "$results/two_expert_calibration_scores_margin_${name}.npz" \
    --max-normal-episode-fp 0 \
    --out "$results/two_expert_fusion_margin_${name}_budget0.json"
  "$py" evaluate_two_expert_fusion.py \
    --scores "$results/two_expert_calibration_scores_margin_${name}.npz" \
    --config "$results/two_expert_fusion_margin_${name}_budget0.json" \
    --out "$results/two_expert_calibration_evaluation_margin_${name}_budget0.json"
}
run_candidate gentle_m1 1.0 0.10 0.025
run_candidate abnormal_only_m2 2.0 0.10 0.0
