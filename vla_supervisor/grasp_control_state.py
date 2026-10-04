"""Selective causal state machine for visual gripper-object control evidence."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Hashable, Iterable


class GraspControlState(str, Enum):
    OPEN = "open"
    WAITING_FOR_EFFECT = "waiting_for_effect"
    CONTROL_CANDIDATE = "control_candidate"
    MOVED_NOT_CONTROLLED = "moved_not_controlled"
    CONTROLLED = "controlled"
    CONTROL_LOST = "control_lost"
    GRASP_NOT_ESTABLISHED = "grasp_not_established"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class GraspControlResult:
    state: GraspControlState
    controlled_object_id: Hashable | None
    informative_motion_frames: int
    confirmation_streak: int
    evidence_valid: bool


class GraspControlMonitor:
    """Fuse closing, proximity, independent motion and gripper synchrony.

    The monitor deliberately does not turn `grasp_not_established` into a
    collision claim. Force/current or abnormal robot response is required by a
    higher-level fusion layer before escalating to possible obstruction.
    """

    def __init__(
        self,
        *,
        close_command_threshold: float = 0.5,
        open_command_threshold: float = -0.5,
        control_confirmation_frames: int = 3,
        effect_grace_motion_frames: int = 20,
        control_loss_motion_frames: int = 3,
    ) -> None:
        if control_confirmation_frames < 1 or effect_grace_motion_frames < 1 or control_loss_motion_frames < 1:
            raise ValueError("all persistence parameters must be positive")
        if open_command_threshold >= close_command_threshold:
            raise ValueError("open threshold must be below close threshold")
        self.close_threshold = close_command_threshold
        self.open_threshold = open_command_threshold
        self.confirmation_required = control_confirmation_frames
        self.grace_frames = effect_grace_motion_frames
        self.loss_required = control_loss_motion_frames
        self.reset()

    def reset(self) -> None:
        self._closing = False
        self._informative_frames = 0
        self._candidate_id: Hashable | None = None
        self._confirmation_streak = 0
        self._controlled_id: Hashable | None = None
        self._loss_streak = 0
        self._state = GraspControlState.OPEN

    def _result(self, state: GraspControlState, valid: bool = True) -> GraspControlResult:
        self._state = state
        return GraspControlResult(
            state, self._controlled_id, self._informative_frames,
            self._confirmation_streak, valid,
        )

    def update(
        self,
        *,
        gripper_command: float,
        gripper_moving: bool,
        nearby_candidate_ids: Iterable[Hashable],
        independently_moving_ids: Iterable[Hashable],
        synchronized_with_gripper_ids: Iterable[Hashable],
        evidence_valid: bool = True,
    ) -> GraspControlResult:
        if gripper_command <= self.open_threshold:
            self.reset()
            return self._result(GraspControlState.OPEN)
        closing_now = gripper_command >= self.close_threshold
        if closing_now and not self._closing:
            self._closing = True
            self._informative_frames = 0
            self._candidate_id = None
            self._confirmation_streak = 0
            self._controlled_id = None
            self._loss_streak = 0
        if not self._closing:
            return self._result(GraspControlState.OPEN)
        if not evidence_valid:
            return self._result(GraspControlState.UNKNOWN, valid=False)

        nearby = set(nearby_candidate_ids)
        moving = set(independently_moving_ids)
        synchronized = set(synchronized_with_gripper_ids)
        controlled_candidates = nearby & moving & synchronized

        if not gripper_moving:
            # A held object may be stationary whenever the end effector is stationary.
            return self._result(
                GraspControlState.CONTROLLED if self._controlled_id is not None
                else GraspControlState.WAITING_FOR_EFFECT
            )
        self._informative_frames += 1

        if self._controlled_id is not None:
            if self._controlled_id in controlled_candidates:
                self._loss_streak = 0
                return self._result(GraspControlState.CONTROLLED)
            self._loss_streak += 1
            if self._loss_streak >= self.loss_required:
                return self._result(GraspControlState.CONTROL_LOST)
            return self._result(GraspControlState.CONTROLLED)

        if controlled_candidates:
            # Ambiguous simultaneous candidates do not build identity confidence.
            if len(controlled_candidates) == 1:
                candidate = next(iter(controlled_candidates))
                if candidate == self._candidate_id:
                    self._confirmation_streak += 1
                else:
                    self._candidate_id = candidate
                    self._confirmation_streak = 1
                if self._confirmation_streak >= self.confirmation_required:
                    self._controlled_id = candidate
                    return self._result(GraspControlState.CONTROLLED)
                return self._result(GraspControlState.CONTROL_CANDIDATE)
            else:
                self._candidate_id = None
                self._confirmation_streak = 0
        else:
            self._candidate_id = None
            self._confirmation_streak = 0

        if self._informative_frames >= self.grace_frames:
            return self._result(GraspControlState.GRASP_NOT_ESTABLISHED)
        if nearby & moving:
            return self._result(GraspControlState.MOVED_NOT_CONTROLLED)
        return self._result(GraspControlState.WAITING_FOR_EFFECT)
