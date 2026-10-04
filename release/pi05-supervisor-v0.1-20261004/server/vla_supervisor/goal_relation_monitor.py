"""Task-agnostic temporal monitor for task-goal relation evidence.

Perception adapters produce RelationEvidence.  This module only combines
evidence over time and deliberately knows nothing about LIBERO task IDs or
simulator ground-truth state.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from statistics import median
from typing import Iterable


class TruthState(str, Enum):
    SATISFIED = "satisfied"
    VIOLATED = "violated"
    UNCERTAIN = "uncertain"
    UNOBSERVABLE = "unobservable"


class GoalEvent(str, Enum):
    CONTINUE = "continue"
    COMPLETE = "complete"
    GOAL_NO_PROGRESS = "goal_no_progress"
    GOAL_REGRESSION = "goal_regression"
    NEED_REOBSERVATION = "need_reobservation"


@dataclass(frozen=True)
class RelationEvidence:
    goal_id: str
    predicate: str
    state: TruthState
    progress: float | None
    confidence: float
    timestamp: float
    uncertainty_reason: str | None = None
    provenance: str = "unknown"

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be within [0, 1]")
        if self.progress is not None and not 0.0 <= self.progress <= 1.0:
            raise ValueError("progress must be within [0, 1]")


@dataclass
class GoalRelationMonitor:
    goal_ids: tuple[str, ...]
    confirmation_frames: int = 3
    no_progress_window: int = 5
    min_progress_gain: float = 0.03
    min_confidence: float = 0.60
    unknown_patience: int = 3
    _history: dict[str, deque[RelationEvidence]] = field(init=False)
    _ever_confirmed: set[str] = field(default_factory=set, init=False)

    def __post_init__(self) -> None:
        if not self.goal_ids:
            raise ValueError("at least one goal is required")
        if self.confirmation_frames < 1 or self.no_progress_window < 2:
            raise ValueError("invalid temporal window")
        size = max(self.confirmation_frames, self.no_progress_window, self.unknown_patience)
        self._history = {goal_id: deque(maxlen=size) for goal_id in self.goal_ids}

    def update(self, evidence: Iterable[RelationEvidence]) -> GoalEvent:
        batch = {item.goal_id: item for item in evidence}
        unknown = set(batch) - set(self.goal_ids)
        if unknown:
            raise KeyError(f"evidence for undeclared goals: {sorted(unknown)}")
        for goal_id, item in batch.items():
            self._history[goal_id].append(item)

        # A previously confirmed predicate becoming confidently violated is
        # stronger than ordinary no-progress (e.g. object fell after release).
        for goal_id in self.goal_ids:
            latest = self._latest_reliable(goal_id)
            if goal_id in self._ever_confirmed and latest and latest.state == TruthState.VIOLATED:
                return GoalEvent.GOAL_REGRESSION

        confirmed = [self._confirmed(goal_id) for goal_id in self.goal_ids]
        for goal_id, yes in zip(self.goal_ids, confirmed):
            if yes:
                self._ever_confirmed.add(goal_id)
        if all(confirmed):
            return GoalEvent.COMPLETE

        if self._all_recent_unknown():
            return GoalEvent.NEED_REOBSERVATION

        # Only report no-progress when every unfinished, observable goal has a
        # sufficiently long reliable history. Unknown evidence never becomes a
        # negative task judgment by accident.
        unfinished = [g for g, done in zip(self.goal_ids, confirmed) if not done]
        if unfinished and all(self._stalled(g) for g in unfinished):
            return GoalEvent.GOAL_NO_PROGRESS
        return GoalEvent.CONTINUE

    def _reliable(self, goal_id: str) -> list[RelationEvidence]:
        return [
            e for e in self._history[goal_id]
            if e.confidence >= self.min_confidence
            and e.state not in {TruthState.UNCERTAIN, TruthState.UNOBSERVABLE}
        ]

    def _latest_reliable(self, goal_id: str) -> RelationEvidence | None:
        reliable = self._reliable(goal_id)
        return reliable[-1] if reliable else None

    def _confirmed(self, goal_id: str) -> bool:
        recent = list(self._history[goal_id])[-self.confirmation_frames :]
        return len(recent) == self.confirmation_frames and all(
            e.confidence >= self.min_confidence and e.state == TruthState.SATISFIED
            for e in recent
        )

    def _all_recent_unknown(self) -> bool:
        for goal_id in self.goal_ids:
            recent = list(self._history[goal_id])[-self.unknown_patience :]
            if len(recent) < self.unknown_patience:
                return False
            if any(
                e.confidence >= self.min_confidence
                and e.state not in {TruthState.UNCERTAIN, TruthState.UNOBSERVABLE}
                for e in recent
            ):
                return False
        return True

    def _stalled(self, goal_id: str) -> bool:
        reliable = [e for e in self._reliable(goal_id) if e.progress is not None]
        recent = reliable[-self.no_progress_window :]
        if len(recent) < self.no_progress_window:
            return False
        split = max(1, len(recent) // 2)
        early = median(e.progress for e in recent[:split] if e.progress is not None)
        late = median(e.progress for e in recent[-split:] if e.progress is not None)
        return late - early < self.min_progress_gain
