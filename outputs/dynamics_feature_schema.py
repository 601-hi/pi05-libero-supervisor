#!/usr/bin/env python3
"""Canonical deployable condition-feature schema for both dynamics experts."""
from __future__ import annotations

import hashlib
import json
from typing import Mapping, Sequence

import numpy as np


DT = 0.05
EPS = 1e-12
ACTION_DIM = 7
CONDITION_DIM_WITH_TASK = 68

FEATURE_GROUPS = (
    ("target_translation_m", 3),
    ("target_translation_norm_m", 1),
    ("rotation_action", 3),
    ("gripper_action", 1),
    ("joint_position", 7),
    ("joint_velocity", 7),
    ("eef_position_m", 3),
    ("eef_quaternion", 4),
    ("gripper_position", 2),
    ("gripper_velocity", 2),
    ("previous_intended_action", 7),
    ("previous_actual_translation_m", 3),
    ("intended_action_delta", 7),
    ("command_turn_cosine", 1),
    ("velocity_alignment_cosine", 1),
    ("previous_eef_speed_mps", 1),
    ("chunk_phase_onehot", 5),
    ("task_onehot", 10),
)


def schema_sha256() -> str:
    payload = json.dumps(FEATURE_GROUPS, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _array(row: Mapping[str, object], key: str, expected: int) -> np.ndarray:
    value = np.asarray(row[key], dtype=np.float64)
    if value.shape != (expected,):
        raise ValueError(f"{key} must have shape ({expected},), got {value.shape}")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{key} contains non-finite values")
    return value


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + EPS))


def build_condition_feature(
    step_row: Mapping[str, object],
    previous_intended_action: Sequence[float],
    previous_actual_translation_m: Sequence[float],
    task_id: int,
    replan_steps: int = 5,
) -> np.ndarray:
    """Build the exact pre-step condition vector used by the frozen normal expert.

    `step_row` may contain post-step fields because trace records are self-contained,
    but this function deliberately reads only intended action and before-state fields.
    """
    if not 0 <= task_id < 10:
        raise ValueError(f"task_id must be in [0,9], got {task_id}")
    if replan_steps != 5:
        raise ValueError("the frozen expert was trained with replan_steps=5")
    action = _array(step_row, "intended_action", ACTION_DIM)
    target = _array(step_row, "intended_target_translation", 3)
    previous_action = np.asarray(previous_intended_action, dtype=np.float64)
    previous_actual = np.asarray(previous_actual_translation_m, dtype=np.float64)
    if previous_action.shape != (ACTION_DIM,):
        raise ValueError(f"previous_intended_action must have shape (7,), got {previous_action.shape}")
    if previous_actual.shape != (3,):
        raise ValueError(f"previous_actual_translation_m must have shape (3,), got {previous_actual.shape}")
    phase = int(step_row["action_index"]) % replan_steps
    phase_onehot = np.eye(replan_steps, dtype=np.float64)[phase]
    task_onehot = np.eye(10, dtype=np.float64)[task_id]
    feature = np.r_[
        target,
        np.linalg.norm(target),
        action[3:6],
        action[6],
        _array(step_row, "joint_pos_before", 7),
        _array(step_row, "joint_vel_before", 7),
        _array(step_row, "eef_pos_before", 3),
        _array(step_row, "eef_quat_before", 4),
        _array(step_row, "gripper_qpos_before", 2),
        _array(step_row, "gripper_qvel_before", 2),
        previous_action,
        previous_actual,
        action - previous_action,
        _cosine(action[:3], previous_action[:3]),
        _cosine(target, previous_actual),
        np.linalg.norm(previous_actual) / DT,
        phase_onehot,
        task_onehot,
    ]
    if feature.shape != (CONDITION_DIM_WITH_TASK,):
        raise AssertionError(f"internal schema error: expected 68 features, got {feature.shape}")
    return feature


def schema_manifest() -> dict:
    offset = 0
    groups = []
    for name, width in FEATURE_GROUPS:
        groups.append({"name": name, "start": offset, "stop_exclusive": offset + width, "width": width})
        offset += width
    return {"dimension": offset, "groups": groups, "sha256": schema_sha256(), "replan_steps": 5, "dt_seconds": DT}


if __name__ == "__main__":
    print(json.dumps(schema_manifest(), ensure_ascii=False, indent=2))
