#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/gpufree-data/supervisor-tools-v1
PY=/root/gpufree-data/vision-env/bin/python
cd "$ROOT"

sha256sum -c <<'EOF'
e120eeb06255d65b795d6e34ce5711bab986cb3467dcdfb426a9bd250b3f180c  vla_supervisor/controlled_object_selector_v2.py
29b607a217b9ee3f542fb1881fccd4f5c6ee20b59767aaee03d8502d974e7cca  cross_suite_generalization/replay_controlled_object_selector_v2.py
18d26d1e486c7ef4f0881d205f7c3528d20f785bb4f7fc43b3815eeeef26c1c1  cross_suite_generalization/diagnose_identity_v2_features.py
fb9cec4693675272306882792970181f9e9f549b9a53b935ef8e4d3d5dd65ced  cross_suite_generalization/run_causal_identity_holdout_gpu.py
6494e9d4d6289963788d864bc2fe4564e48268a2cba80f98f7357a941d19ab89  outputs/GRIPPER_PROJECTION_WAVE1_FIT.json
c1a149aa6a54720e29a0032dc90676e2fc5fbd9d93eee603ef2abce70f758993  outputs/CAUSAL_IDENTITY_HOLDOUT_PUBLIC_V1.json
990c3fcc5a814fd99d68adb1fed726514690068d8bd1ccdc7e06df81c201d4e2  outputs/CAUSAL_IDENTITY_HOLDOUT_PRIVATE_AUDIT_V1.json
EOF

"$PY" -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; print(torch.cuda.get_device_name(0))'
test ! -e outputs/causal_identity_holdout_wave2/predictions.json
export HF_HOME=/root/gpufree-data/hf-cache
export HF_HUB_OFFLINE=1

"$PY" cross_suite_generalization/run_causal_identity_holdout_gpu.py \
  --public-manifest outputs/CAUSAL_IDENTITY_HOLDOUT_PUBLIC_V1.json \
  --private-map outputs/CAUSAL_IDENTITY_HOLDOUT_PRIVATE_AUDIT_V1.json \
  --output-dir outputs/causal_identity_holdout_wave2 \
  --wave 2 --family rigid_object_transport

"$PY" cross_suite_generalization/export_initial_candidate_masks_gpu.py \
  --public outputs/CAUSAL_IDENTITY_HOLDOUT_PUBLIC_V1.json \
  --private outputs/CAUSAL_IDENTITY_HOLDOUT_PRIVATE_AUDIT_V1.json \
  --predictions outputs/causal_identity_holdout_wave2/predictions.json \
  --output-dir outputs/causal_identity_wave2_initial_masks

echo WAVE2_GPU_STAGE_COMPLETE
