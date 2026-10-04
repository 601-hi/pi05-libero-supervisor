"""Task-agnostic routing from diagnostic evidence to recovery milestones.

The router does not identify task objects and does not execute motion.  It
expresses *which precondition must be restored*.  Physical bridge actions stay
gated by the independent intervention planner.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable

from .events import MonitorEvent


class RecoveryTarget(str, Enum):
    REQUEST_MORE_EVIDENCE = "request_more_evidence"
    REACQUIRE_INTENDED_TARGET = "reacquire_intended_target"
    ESTABLISH_STABLE_CONTROL = "establish_stable_control"
    RELEASE_WRONG_OBJECT = "release_wrong_object"
    CLEAR_OBSTRUCTION = "clear_obstruction"
    RESTORE_GOAL_RELATION = "restore_goal_relation"


@dataclass(frozen=True)
class RecoveryTargetDecision:
    target: RecoveryTarget
    confidence: float
    evidence_sources: tuple[str, ...]
    prompt_suffix: str
    actionable: bool


class RecoveryTargetRouter:
    """Conservative evidence router with no LIBERO-specific object rules."""

    def __init__(self, actionable_confidence: float = 0.75):
        if not 0.0 <= actionable_confidence <= 1.0:
            raise ValueError("actionable_confidence must be in [0, 1]")
        self.actionable_confidence = actionable_confidence

    @staticmethod
    def _matches(events: tuple[MonitorEvent, ...], states: set[str]):
        return tuple(e for e in events if str(e.evidence.get("diagnostic_state")) in states)

    def _decision(self, target, matches, prompt):
        confidence = max((float(e.confidence) for e in matches), default=0.0)
        sources = tuple(sorted({str(e.source) for e in matches}))
        return RecoveryTargetDecision(
            target, confidence, sources, prompt,
            bool(matches) and confidence >= self.actionable_confidence,
        )

    def route(self, events: Iterable[MonitorEvent]) -> RecoveryTargetDecision:
        events = tuple(events)
        rules = (
            (
                RecoveryTarget.RELEASE_WRONG_OBJECT,
                {"wrong_object_control"},
                "Release into a safe area, re-observe the intended target, and restart target acquisition.",
            ),
            (
                RecoveryTarget.CLEAR_OBSTRUCTION,
                {"fixed_obstacle_or_jam"},
                "Stop pushing, move clear along a verified safe direction, and reacquire a free approach.",
            ),
            (
                RecoveryTarget.REACQUIRE_INTENDED_TARGET,
                {"object_loss_risk", "target_lost", "approach_progress_stalled"},
                "Re-observe the intended target at its current location and reacquire it before transport.",
            ),
            (
                RecoveryTarget.ESTABLISH_STABLE_CONTROL,
                {"empty_grasp_or_miss", "unstable_object_control", "grasp_progress_stalled"},
                "Return to the grasp stage and establish stable control of the intended target.",
            ),
            (
                RecoveryTarget.RESTORE_GOAL_RELATION,
                {"goal_relation_not_satisfied", "placement_failed", "goal_relation_progress_stalled"},
                "Keep or safely reacquire the intended target and restore the required goal relation.",
            ),
        )
        for target, states, prompt in rules:
            matches = self._matches(events, states)
            if matches:
                return self._decision(target, matches, prompt)
        return RecoveryTargetDecision(
            RecoveryTarget.REQUEST_MORE_EVIDENCE,
            max((float(e.confidence) for e in events), default=0.0),
            tuple(sorted({str(e.source) for e in events})),
            "Hold safely and gather object-control, contact, and goal-progress evidence before choosing a recovery milestone.",
            False,
        )
