"""Dataset-independent causal transition contract for dynamics supervision."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np


ALLOWED_COMMAND_MODES = {
    "eef_delta_pose",
    "eef_velocity",
    "eef_absolute_pose",
    "joint_delta_position",
    "joint_velocity",
    "joint_absolute_position",
}
FORBIDDEN_FEATURE_TOKENS = ("oracle_only_", "reward", "success", "contact", "object_pose")


@dataclass(frozen=True)
class CommandSemantics:
    mode: str
    reference_frame: str
    units: str
    rotation_representation: str
    gripper_semantics: str


@dataclass(frozen=True)
class CausalTransition:
    dataset: str
    episode_id: str
    step_id: int
    robot_model: str
    controller_mode: str
    command_source: str
    timing_source: str
    command_timestamp: float
    state_before_timestamp: float
    state_after_timestamp: float
    command: np.ndarray
    command_semantics: CommandSemantics
    state_before: Mapping[str, Any]
    state_after: Mapping[str, Any]
    optional_sensors: Mapping[str, Any] = field(default_factory=dict)
    quality_flags: Mapping[str, bool] = field(default_factory=dict)

    @property
    def delta_t_seconds(self) -> float:
        return float(self.state_after_timestamp - self.state_before_timestamp)


def validate_transition(row: CausalTransition) -> list[str]:
    """Return all contract violations instead of silently coercing a sample."""
    errors: list[str] = []
    if row.command_semantics.mode not in ALLOWED_COMMAND_MODES:
        errors.append(f"unsupported command mode: {row.command_semantics.mode!r}")
    for name, value in {
        "dataset": row.dataset,
        "episode_id": row.episode_id,
        "robot_model": row.robot_model,
        "controller_mode": row.controller_mode,
        "command_source": row.command_source,
        "timing_source": row.timing_source,
        "reference_frame": row.command_semantics.reference_frame,
        "units": row.command_semantics.units,
        "gripper_semantics": row.command_semantics.gripper_semantics,
    }.items():
        if not str(value).strip():
            errors.append(f"missing {name}")
    command = np.asarray(row.command)
    if command.ndim != 1 or command.size == 0 or not np.isfinite(command).all():
        errors.append("command must be a finite non-empty vector")
    if row.step_id < 0:
        errors.append("step_id must be non-negative")
    if row.command_timestamp > row.state_after_timestamp:
        errors.append("command timestamp is later than the measured response")
    if row.delta_t_seconds <= 0.0:
        errors.append("state_after_timestamp must be later than state_before_timestamp")
    if not row.state_before or not row.state_after:
        errors.append("both measured before and after states are required")
    deployable_keys = set(row.state_before) | set(row.state_after) | set(row.optional_sensors)
    for key in sorted(deployable_keys):
        lowered = key.lower()
        if any(token in lowered for token in FORBIDDEN_FEATURE_TOKENS):
            errors.append(f"forbidden deployable feature key: {key}")
    failed_quality = sorted(name for name, passed in row.quality_flags.items() if not passed)
    if failed_quality:
        errors.append(f"failed quality flags: {failed_quality}")
    return errors


def require_valid_transition(row: CausalTransition) -> CausalTransition:
    errors = validate_transition(row)
    if errors:
        raise ValueError("; ".join(errors))
    return row
