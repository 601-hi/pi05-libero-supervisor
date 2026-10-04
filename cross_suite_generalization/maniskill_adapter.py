"""Adapter for decoded ManiSkill demonstration arrays without importing ManiSkill."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from common_dynamics_transition import CausalTransition, CommandSemantics, require_valid_transition


CONTROL_MODE_MAP = {
    "pd_ee_delta_pose": ("eef_delta_pose", "robot_base", "normalized controller input"),
    "pd_joint_delta_pos": ("joint_delta_position", "joint", "normalized controller input"),
    "pd_joint_pos": ("joint_absolute_position", "joint", "normalized controller input"),
}


@dataclass(frozen=True)
class ManiSkillConfig:
    control_mode: str
    control_frequency_hz: float
    robot_model: str
    command_semantics_verified: bool = False
    units_verified: bool = False
    timing_source: str = "fixed_simulation_control_cadence"


def adapt_trajectory(
    actions: np.ndarray,
    measured_states: Mapping[str, np.ndarray],
    *,
    episode_id: str,
    config: ManiSkillConfig,
) -> list[CausalTransition]:
    actions = np.asarray(actions, dtype=np.float64)
    if actions.ndim != 2:
        raise ValueError("actions must have shape [T, A]")
    if config.control_mode not in CONTROL_MODE_MAP:
        raise ValueError(f"unsupported or ambiguous ManiSkill control mode: {config.control_mode!r}")
    if config.control_frequency_hz <= 0.0:
        raise ValueError("control_frequency_hz must be positive")
    if not measured_states:
        raise ValueError("measured_states cannot be empty")
    states = {name: np.asarray(value) for name, value in measured_states.items()}
    for name, value in states.items():
        if value.shape[0] != actions.shape[0] + 1:
            raise ValueError(f"state {name} must contain T+1 rows")
        if not np.issubdtype(value.dtype, np.number) or not np.isfinite(value).all():
            raise ValueError(f"state {name} must be finite numeric data")
    mode, frame, units = CONTROL_MODE_MAP[config.control_mode]
    dt = 1.0 / config.control_frequency_hz
    transitions = []
    for index, command in enumerate(actions):
        before_time = index * dt
        row = CausalTransition(
            dataset="ManiSkill",
            episode_id=str(episode_id),
            step_id=index,
            robot_model=config.robot_model,
            controller_mode=config.control_mode,
            command_source="recorded_trajectory.actions",
            timing_source=config.timing_source,
            command_timestamp=before_time,
            state_before_timestamp=before_time,
            state_after_timestamp=before_time + dt,
            command=command,
            command_semantics=CommandSemantics(
                mode=mode,
                reference_frame=frame,
                units=units,
                rotation_representation="controller_mode_specific",
                gripper_semantics="controller_mode_specific_last_dimension",
            ),
            state_before={name: value[index] for name, value in states.items()},
            state_after={name: value[index + 1] for name, value in states.items()},
            quality_flags={
                "timestamp_monotonic": True,
                "alignment_tolerance_satisfied": True,
                "no_dropped_state_transition": True,
                "command_semantics_verified_from_primary_source": config.command_semantics_verified,
                "units_verified": config.units_verified,
                "no_oracle_label_in_features": True,
            },
        )
        transitions.append(require_valid_transition(row))
    return transitions
