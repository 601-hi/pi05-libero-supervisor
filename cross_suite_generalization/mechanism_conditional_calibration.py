#!/usr/bin/env python3
"""Pre-action mechanism bins and robust normal-only score standardization."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MechanismThresholds:
    command_large_m: float = 0.035
    joint_slow_rad_s: float = 0.25
    joint_margin_near: float = 0.10
    gripper_moving_m_s: float = 0.005


def assign_regime(command_translation_norm, joint_velocity_norm, joint_limit_margin,
                  gripper_velocity_norm, thresholds=MechanismThresholds()):
    """Assign one of four causal/pre-action regimes; lower integer wins."""
    command_translation_norm = np.asarray(command_translation_norm)
    joint_velocity_norm = np.asarray(joint_velocity_norm)
    joint_limit_margin = np.asarray(joint_limit_margin)
    gripper_velocity_norm = np.asarray(gripper_velocity_norm)
    large = command_translation_norm >= thresholds.command_large_m
    slow = joint_velocity_norm < thresholds.joint_slow_rad_s
    near = joint_limit_margin < thresholds.joint_margin_near
    gripper_moving = gripper_velocity_norm > thresholds.gripper_moving_m_s
    special = near | (large & slow) | (large & gripper_moving)
    # 0 special constraint/transition, 1 regular large command,
    # 2 other slow state, 3 other regular state.
    return np.select([special, large, slow], [0, 1, 2], default=3).astype(np.int8)


def fit_normal_standardizer(scores, regimes, min_group_samples=20):
    """Fit median/IQR using explicitly confirmed normal samples only."""
    scores = np.asarray(scores, dtype=float)
    regimes = np.asarray(regimes)
    center_global = float(np.median(scores))
    scale_global = max(float(np.percentile(scores, 75) - np.percentile(scores, 25)), 1e-3)
    params = {}
    for group in range(4):
        values = scores[regimes == group]
        if len(values) < min_group_samples:
            params[group] = (center_global, scale_global, int(len(values)), True)
        else:
            center = float(np.median(values))
            scale = max(float(np.percentile(values, 75) - np.percentile(values, 25)), 1e-3)
            params[group] = (center, scale, int(len(values)), False)
    return params


def apply_normal_standardizer(scores, regimes, params):
    scores = np.asarray(scores, dtype=float)
    regimes = np.asarray(regimes)
    result = np.empty_like(scores)
    for group, (center, scale, _, _) in params.items():
        mask = regimes == group
        result[mask] = (scores[mask] - center) / scale
    return result
