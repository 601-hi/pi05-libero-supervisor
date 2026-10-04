"""Progress-conditioned soft budgets for long-horizon rollback phases."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BudgetDecision:
    allow: bool
    extended: bool
    effective_limit: int
    reason: str
    recent_net_progress_m: float | None = None
    recent_slope_m_per_step: float | None = None


class AdaptivePhaseBudget:
    """Extend a phase only while closed-loop target error keeps improving.

    ``soft_limit`` is the first review point, not an unconditional stop.
    Extensions are granted in bounded chunks and can never exceed
    ``hard_limit``.  Collision and other safety gates remain external and
    retain priority over this liveness mechanism.
    """

    def __init__(
        self,
        *,
        soft_limit: int,
        hard_limit: int,
        extension_chunk: int = 80,
        window: int = 20,
        minimum_net_progress_m: float = 0.0005,
        minimum_slope_m_per_step: float = 0.00001,
        require_target_advance: bool = False,
    ):
        if not 0 < soft_limit <= hard_limit:
            raise ValueError("require 0 < soft_limit <= hard_limit")
        if extension_chunk <= 0 or window < 3:
            raise ValueError("extension_chunk must be positive and window >= 3")
        self.soft_limit = int(soft_limit)
        self.hard_limit = int(hard_limit)
        self.extension_chunk = int(extension_chunk)
        self.window = int(window)
        self.minimum_net_progress_m = float(minimum_net_progress_m)
        self.minimum_slope_m_per_step = float(minimum_slope_m_per_step)
        self.require_target_advance = bool(require_target_advance)
        self.reset()

    def reset(self) -> None:
        self.effective_limit = self.soft_limit
        self._errors = deque(maxlen=self.window)
        self._target_id = None
        self._target_advances_since_review = 0

    def observe(self, error_m: float | None, *, target_id=None) -> None:
        if error_m is None or not np.isfinite(float(error_m)):
            return
        # A replay reference change creates a new local control problem.  Do
        # not compare errors measured against two different targets.
        if target_id != self._target_id:
            if self._target_id is not None:
                self._target_advances_since_review += 1
            self._errors.clear()
            self._target_id = target_id
        self._errors.append(float(error_m))

    def decide(self, steps: int) -> BudgetDecision:
        steps = int(steps)
        if steps < self.effective_limit:
            return BudgetDecision(True, False, self.effective_limit, "within_budget")
        if steps >= self.hard_limit:
            return BudgetDecision(False, False, self.effective_limit, "hard_limit")
        if self.require_target_advance and self._target_advances_since_review < 1:
            return BudgetDecision(
                False, False, self.effective_limit, "no_target_advance")
        if len(self._errors) < self.window:
            return BudgetDecision(False, False, self.effective_limit, "insufficient_progress_history")

        values = np.asarray(self._errors, dtype=float)
        net_progress = float(values[0] - values[-1])
        slope = float(np.polyfit(np.arange(values.size), values, 1)[0])
        improving = (
            net_progress >= self.minimum_net_progress_m
            and slope <= -self.minimum_slope_m_per_step
        )
        if not improving:
            return BudgetDecision(
                False, False, self.effective_limit, "not_converging",
                net_progress, slope,
            )

        self.effective_limit = min(
            self.hard_limit, self.effective_limit + self.extension_chunk
        )
        self._target_advances_since_review = 0
        return BudgetDecision(
            True, True, self.effective_limit, "progress_conditioned_extension",
            net_progress, slope,
        )

