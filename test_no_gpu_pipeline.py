#!/usr/bin/env python3
"""Small dependency-light contract tests for the dynamics-expert pipeline."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np

from dynamics_feature_schema import build_condition_feature, schema_sha256
from two_expert_fusion import TwoExpertFusion


def row(action_index=0):
    return {
        "action_index": action_index, "intended_action": [0.1] * 7,
        "intended_target_translation": [0.005] * 3,
        "joint_pos_before": [0.0] * 7, "joint_vel_before": [0.0] * 7,
        "eef_pos_before": [0.0] * 3, "eef_quat_before": [1.0, 0.0, 0.0, 0.0],
        "gripper_qpos_before": [0.0] * 2, "gripper_qvel_before": [0.0] * 2,
    }


def main():
    base = row(); feature = build_condition_feature(base, np.zeros(7), np.zeros(3), 2)
    assert feature.shape == (68,) and np.isfinite(feature).all()
    # Post-step, simulator-only, and label fields must have no influence on x.
    contaminated = dict(base)
    contaminated.update({"executed_action": [99] * 7, "actual_translation": [99] * 3,
                         "reward": 99, "success": True, "disturbance_active": True,
                         "translation_action_scale": 0.25, "eef_pos_after": [99] * 3})
    assert np.array_equal(feature, build_condition_feature(contaminated, np.zeros(7), np.zeros(3), 2))
    config = {"normal_absolute_floor": -5.0, "abnormal_absolute_floor": -5.0,
              "ratio_abnormal_threshold": 1.0, "persistence_required": 2, "persistence_window": 3}
    fusion = TwoExpertFusion(config)
    assert fusion.update(0, -10).state == "normal"
    assert fusion.update(-10, 0).state == "known_abnormal"
    assert fusion.update(-10, -10).state == "unknown_abnormal"
    assert fusion.update(0, 0).state == "ambiguous_overlap"
    print(json.dumps({"status": "PASS", "feature_dimension": 68,
                      "feature_schema_sha256": schema_sha256(),
                      "leakage_invariance": True, "fusion_states": 4}, indent=2))


if __name__ == "__main__": main()
