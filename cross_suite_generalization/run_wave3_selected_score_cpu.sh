#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/gpufree-data/supervisor-tools-v1
PY=/root/gpufree-data/vla-workspace/openpi/.venv/bin/python
cd "$ROOT/cross_suite_generalization"
"$PY" score_selected_target_ensemble.py \
  --model ../outputs/remote_results/reliability_sequence_v1/conditional_relational_pilot_seed202609121.pt \
  --model ../outputs/remote_results/reliability_sequence_v1/conditional_relational_pilot_seed202609122.pt \
  --model ../outputs/remote_results/reliability_sequence_v1/conditional_relational_pilot_seed202609123.pt \
  --model ../outputs/remote_results/reliability_sequence_v1/conditional_relational_pilot_seed202609124.pt \
  --model ../outputs/remote_results/reliability_sequence_v1/conditional_relational_pilot_seed202609125.pt \
  --trace /root/gpufree-data/libero-traces/natural_failure_stage_a_seed34/traces/libero_90_task19_seed34_natural_5ep.jsonl \
  --trace /root/gpufree-data/libero-traces/natural_failure_stage_a_seed34/traces/libero_90_task29_seed34_natural_15ep.jsonl \
  --trace /root/gpufree-data/libero-traces/natural_failure_stage_a_seed34/traces/libero_spatial_task0_seed34_natural_10ep.jsonl \
  --trace /root/gpufree-data/libero-traces/natural_failure_stage_a_seed34/traces/libero_spatial_task1_seed34_natural_10ep.jsonl \
  --trace /root/gpufree-data/libero-traces/natural_failure_stage_a_seed34/traces/libero_spatial_task3_seed34_natural_15ep.jsonl \
  --trace /root/gpufree-data/libero-traces/natural_failure_stage_a_seed34/traces/libero_spatial_task4_seed34_natural_15ep.jsonl \
  --trace /root/gpufree-data/libero-traces/natural_failure_stage_a_seed34/traces/libero_spatial_task7_seed34_natural_15ep.jsonl \
  --trace /root/gpufree-data/libero-traces/natural_failure_stage_a_seed34/traces/libero_spatial_task9_seed34_natural_15ep.jsonl \
  --trace-map ../outputs/WAVE3_ACTION_SCORE_TRACE_MAP.json \
  --private-map ../outputs/CAUSAL_IDENTITY_WAVE3_PRIVATE_MAP.json \
  --output ../outputs/WAVE3_ACTION_EXPERT_SCORES_ZERO_FIT.npz
