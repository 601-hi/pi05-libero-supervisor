#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/gpufree-data/supervisor-tools-v1
PY=/root/gpufree-data/vla-workspace/openpi/examples/libero/.venv/bin/python
TRACE_ROOT=/root/gpufree-data/libero-traces/natural_failure_stage_a_seed34/traces
cd "$ROOT"

test -s outputs/causal_identity_holdout_wave2/predictions.json
test -d outputs/causal_identity_wave2_initial_masks

"$PY" cross_suite_generalization/diagnose_identity_v2_features.py \
  --public outputs/CAUSAL_IDENTITY_HOLDOUT_PUBLIC_V1.json \
  --private outputs/CAUSAL_IDENTITY_HOLDOUT_PRIVATE_AUDIT_V1.json \
  --predictions outputs/causal_identity_holdout_wave2/predictions.json \
  --trace-root "$TRACE_ROOT" \
  --output outputs/CAUSAL_IDENTITY_WAVE2_V2_FEATURES.json

"$PY" cross_suite_generalization/export_wave1_robot_state.py \
  --private outputs/CAUSAL_IDENTITY_HOLDOUT_PRIVATE_AUDIT_V1.json \
  --features outputs/CAUSAL_IDENTITY_WAVE2_V2_FEATURES.json \
  --trace-root "$TRACE_ROOT" \
  --output outputs/CAUSAL_IDENTITY_WAVE2_ROBOT_STATE.json

"$PY" cross_suite_generalization/replay_controlled_object_selector_v2.py \
  --predictions outputs/causal_identity_holdout_wave2/predictions.json \
  --features outputs/CAUSAL_IDENTITY_WAVE2_V2_FEATURES.json \
  --robot-state outputs/CAUSAL_IDENTITY_WAVE2_ROBOT_STATE.json \
  --projection outputs/GRIPPER_PROJECTION_WAVE1_FIT.json \
  --output outputs/CAUSAL_IDENTITY_WAVE2_V2_REPLAY_UNLABELED.json

echo WAVE2_CPU_BLIND_PREDICTION_COMPLETE
