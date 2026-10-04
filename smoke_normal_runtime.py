#!/usr/bin/env python3
"""Numerical loading smoke for the frozen normal expert."""
import json
import sys

import numpy as np

from normal_dynamics_expert_runtime import NormalDynamicsExpertRuntime, checkpoint_summary


checkpoint = sys.argv[1]
runtime = NormalDynamicsExpertRuntime(checkpoint, profile="90")
result = runtime.update(np.zeros(68), np.zeros(3), task_id=0, intended_target_norm_m=0.0)
assert result.predicted_translation_m.shape == (3,)
assert result.predictive_std_m.shape == (3,)
assert np.isfinite(result.score)
print(json.dumps({"status": "PASS", "checkpoint": checkpoint,
                  "summary": checkpoint_summary(checkpoint),
                  "zero_feature_score_finite": True}, indent=2))
