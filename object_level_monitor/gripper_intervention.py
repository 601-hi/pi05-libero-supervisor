"""Controlled gripper-channel interventions for paired causal rollouts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


VALID_MODES = {"none", "force_open", "force_closed", "delay"}


@dataclass(frozen=True)
class GripperInterventionResult:
    executed_action: np.ndarray
    applied: bool
    source_history_offset: int | None


def apply_gripper_intervention(
    intended_action: Sequence[float],
    *,
    active: bool,
    mode: str,
    intended_gripper_history: Sequence[float],
    delay_steps: int = 0,
) -> GripperInterventionResult:
    """Modify only the last (gripper) action coordinate.

    The function is causal. ``delay`` uses a past intended command and never
    peeks at future policy output. Before enough history exists it leaves the
    command unchanged, making the absence of an intervention explicit.
    """

    if mode not in VALID_MODES:
        raise ValueError(f"mode must be one of {sorted(VALID_MODES)}, got {mode!r}")
    if delay_steps < 0:
        raise ValueError("delay_steps must be non-negative")

    intended = np.asarray(intended_action, dtype=float)
    if intended.ndim != 1 or intended.size < 1:
        raise ValueError("intended_action must be a non-empty one-dimensional action")
    executed = intended.copy()
    if not active or mode == "none":
        return GripperInterventionResult(executed, False, None)

    if mode == "force_open":
        executed[-1] = -1.0
        source_offset = None
    elif mode == "force_closed":
        executed[-1] = 1.0
        source_offset = None
    else:
        if delay_steps == 0:
            return GripperInterventionResult(executed, False, 0)
        if len(intended_gripper_history) < delay_steps:
            return GripperInterventionResult(executed, False, None)
        executed[-1] = float(intended_gripper_history[-delay_steps])
        source_offset = delay_steps

    changed = not np.array_equal(executed, intended)
    return GripperInterventionResult(executed, changed, source_offset)
