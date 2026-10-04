"""Causal finite-time monitor for a gripper that fails to close on command."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class MonitorState(str, Enum):
    ARMED = "armed"
    WATCHING = "watching"
    IDLE = "idle_until_open"
    LATCHED = "latched_alarm"


@dataclass(frozen=True)
class MonitorResult:
    state: MonitorState
    alarm: bool
    new_alarm: bool
    observed_closure: float | None = None


class GripperCloseResponseMonitor:
    """Detect insufficient finite-time closure from deployable signals only.

    The monitor starts a horizon when the gripper is sufficiently open and a
    closing command has persisted.  A fault is latched after the decision so a
    single persistent failure cannot be counted repeatedly.  Clearing a fault
    is deliberately an explicit supervisory action via ``reset``.
    """

    def __init__(
        self,
        *,
        open_width_min: float = 0.04,
        close_command_threshold: float = 0.5,
        open_command_threshold: float = -0.5,
        consecutive_close_commands: int = 2,
        consecutive_open_commands_to_rearm: int = 2,
        horizon_steps: int = 3,
        minimum_closure: float = 0.0023790714853767214,
    ) -> None:
        if consecutive_close_commands <= 0 or consecutive_open_commands_to_rearm <= 0:
            raise ValueError("command persistence must be positive")
        if horizon_steps < consecutive_close_commands:
            raise ValueError("horizon must include the close-command trigger")
        self.open_width_min = open_width_min
        self.close_command_threshold = close_command_threshold
        self.open_command_threshold = open_command_threshold
        self.close_required = consecutive_close_commands
        self.open_required = consecutive_open_commands_to_rearm
        self.horizon_steps = horizon_steps
        self.minimum_closure = minimum_closure
        self.reset()

    def reset(self) -> None:
        self.state = MonitorState.ARMED
        self._close_run = 0
        self._open_run = 0
        self._candidate_width: float | None = None
        self._start_width: float | None = None
        self._watch_steps = 0

    def update(self, command: float, width_before: float, width_after: float) -> MonitorResult:
        if self.state is MonitorState.LATCHED:
            return MonitorResult(self.state, alarm=True, new_alarm=False)

        if self.state is MonitorState.IDLE:
            self._open_run = self._open_run + 1 if command <= self.open_command_threshold else 0
            if self._open_run >= self.open_required:
                self.state = MonitorState.ARMED
                self._open_run = 0
            return MonitorResult(self.state, alarm=False, new_alarm=False)

        if self.state is MonitorState.ARMED:
            if command >= self.close_command_threshold:
                if self._close_run == 0:
                    self._candidate_width = width_before
                self._close_run += 1
            else:
                self._close_run = 0
                self._candidate_width = None
            if self._close_run >= self.close_required:
                assert self._candidate_width is not None
                if self._candidate_width >= self.open_width_min:
                    self.state = MonitorState.WATCHING
                    self._start_width = self._candidate_width
                    self._watch_steps = self.close_required
                    if self._watch_steps < self.horizon_steps:
                        return MonitorResult(self.state, alarm=False, new_alarm=False)
                else:
                    self.state = MonitorState.IDLE
                    return MonitorResult(self.state, alarm=False, new_alarm=False)
            else:
                return MonitorResult(self.state, alarm=False, new_alarm=False)

        assert self.state is MonitorState.WATCHING and self._start_width is not None
        # The trigger update already accounts for ``close_required`` steps.
        if self._watch_steps < self.horizon_steps:
            self._watch_steps += 1
        if self._watch_steps < self.horizon_steps:
            return MonitorResult(self.state, alarm=False, new_alarm=False)
        closure = self._start_width - width_after
        if closure < self.minimum_closure:
            self.state = MonitorState.LATCHED
            return MonitorResult(self.state, alarm=True, new_alarm=True, observed_closure=closure)
        self.state = MonitorState.IDLE
        self._close_run = 0
        return MonitorResult(self.state, alarm=False, new_alarm=False, observed_closure=closure)
