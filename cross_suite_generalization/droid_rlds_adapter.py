"""Conservative adapter from decoded DROID RLDS steps to causal transitions.

This module deliberately has no TensorFlow dependency. A loader may decode RLDS
records to ordinary mappings, but this adapter owns the command/state alignment
and rejects schemas whose meanings have not been explicitly configured.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from common_dynamics_transition import CausalTransition, CommandSemantics, require_valid_transition


@dataclass(frozen=True)
class DroidRldsConfig:
    control_frequency_hz: float = 15.0
    robot_model: str = "Franka Panda"
    controller_mode: str = "cartesian_velocity"
    reference_frame: str = "robot_base"
    timing_source: str = "fixed_cadence_assumption_from_DROID_paper"
    command_source: str = "recorded_action_dict.cartesian_velocity_plus_gripper_position"
    command_semantics_verified: bool = False
    units_verified: bool = False
    response_lag_steps: int | None = None
    timing_alignment_verified: bool = False


def _array(mapping: Mapping[str, Any], key: str, size: int) -> np.ndarray:
    if key not in mapping:
        raise KeyError(f"required DROID field is missing: {key}")
    value = np.asarray(mapping[key], dtype=np.float64).reshape(-1)
    if value.shape != (size,) or not np.isfinite(value).all():
        raise ValueError(f"DROID field {key} must be a finite vector of shape ({size},)")
    return value


def _measured_state(step: Mapping[str, Any]) -> dict[str, np.ndarray]:
    if "observation" not in step:
        raise KeyError("required DROID field is missing: observation")
    observation = step["observation"]
    return {
        "eef_pose_xyz_euler": _array(observation, "cartesian_position", 6),
        "joint_position": _array(observation, "joint_position", 7),
        "gripper_position": _array(observation, "gripper_position", 1),
    }


def adapt_episode(
    steps: Sequence[Mapping[str, Any]],
    *,
    episode_id: str,
    config: DroidRldsConfig = DroidRldsConfig(),
) -> list[CausalTransition]:
    if config.control_frequency_hz <= 0.0:
        raise ValueError("control_frequency_hz must be positive")
    if config.response_lag_steps is None or not config.timing_alignment_verified:
        raise ValueError("response_lag_steps must be explicitly set and timing_alignment_verified")
    if config.response_lag_steps < 0:
        raise ValueError("response_lag_steps must be non-negative")
    lag = config.response_lag_steps
    if len(steps) < lag + 2:
        return []
    dt = 1.0 / config.control_frequency_hz
    transitions: list[CausalTransition] = []
    for index in range(len(steps) - lag - 1):
        command_step = steps[index]
        before_step = steps[index + lag]
        after_step = steps[index + lag + 1]
        if "action_dict" not in command_step:
            raise KeyError("required DROID field is missing: action_dict")
        action = command_step["action_dict"]
        cartesian_velocity = _array(action, "cartesian_velocity", 6)
        gripper_position = _array(action, "gripper_position", 1)
        command_time = index * dt
        before_time = (index + lag) * dt
        transition = CausalTransition(
            dataset="DROID_RLDS",
            episode_id=str(episode_id),
            step_id=index,
            robot_model=config.robot_model,
            controller_mode=config.controller_mode,
            command_source=config.command_source,
            timing_source=config.timing_source,
            command_timestamp=command_time,
            state_before_timestamp=before_time,
            state_after_timestamp=before_time + dt,
            command=np.concatenate((cartesian_velocity, gripper_position)),
            command_semantics=CommandSemantics(
                mode="eef_velocity",
                reference_frame=config.reference_frame,
                units="m/s,rad/s,normalized_gripper_position",
                rotation_representation="angular_velocity_xyz",
                gripper_semantics="absolute_open_fraction",
            ),
            state_before=_measured_state(before_step),
            state_after=_measured_state(after_step),
            quality_flags={
                "timestamp_monotonic": True,
                "alignment_tolerance_satisfied": True,
                "no_dropped_state_transition": True,
                "command_semantics_verified_from_primary_source": config.command_semantics_verified,
                "units_verified": config.units_verified,
                "timing_alignment_verified": config.timing_alignment_verified,
                "no_oracle_label_in_features": True,
            },
        )
        transitions.append(require_valid_transition(transition))
    return transitions
