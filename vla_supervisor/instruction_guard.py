from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .events import EventType, MonitorEvent, RecommendedAction


@dataclass(frozen=True)
class GuardConfig:
    # π0.5 outputs can be a few thousandths beyond one before the controller
    # clips them, so the default guard keeps a small numerical tolerance.
    action_abs_limit: float = 1.05
    action_jump_limit: float = 3.0
    joint_position_low: tuple[float, ...] | None = None
    joint_position_high: tuple[float, ...] | None = None
    joint_velocity_abs_limit: float | None = None
    workspace_low: tuple[float, float, float] | None = None
    workspace_high: tuple[float, float, float] | None = None


class InstructionGuard:
    name = "instruction_guard"

    def __init__(self, config: GuardConfig):
        self.config = config

    def check(self, action: np.ndarray, state: Mapping[str, Any], previous_action=None,
              action_index: int | None = None) -> MonitorEvent:
        action = np.asarray(action, float)
        violations = {}
        if not np.isfinite(action).all():
            violations["non_finite_action"] = True
        max_action = float(np.max(np.abs(action)))
        if max_action > self.config.action_abs_limit:
            violations["max_abs_action"] = max_action
        if previous_action is not None:
            jump = float(np.linalg.norm(action - np.asarray(previous_action, float)))
            if jump > self.config.action_jump_limit:
                violations["action_jump_norm"] = jump
        self._check_bounds(state, "joint_pos", self.config.joint_position_low,
                           self.config.joint_position_high, violations)
        if self.config.joint_velocity_abs_limit is not None and "joint_vel" in state:
            velocity = float(np.max(np.abs(np.asarray(state["joint_vel"], float))))
            if velocity > self.config.joint_velocity_abs_limit:
                violations["max_abs_joint_velocity"] = velocity
        self._check_bounds(state, "eef_pos", self.config.workspace_low,
                           self.config.workspace_high, violations)
        unsafe = bool(violations)
        return MonitorEvent(
            EventType.INSTRUCTION_UNSAFE if unsafe else EventType.NORMAL,
            score=float(len(violations)), confidence=1.0, evidence=violations,
            recommended_action=(RecommendedAction.STOP_AND_REPLAN if unsafe else RecommendedAction.CONTINUE),
            source=self.name, action_index=action_index,
        )

    @staticmethod
    def _check_bounds(state, key, low, high, violations):
        if key not in state or low is None or high is None:
            return
        value, lo, hi = map(lambda x: np.asarray(x, float), (state[key], low, high))
        if value.shape != lo.shape or value.shape != hi.shape:
            violations[f"{key}_shape_mismatch"] = True
        elif np.any(value < lo) or np.any(value > hi):
            violations[f"{key}_outside_bounds"] = True
