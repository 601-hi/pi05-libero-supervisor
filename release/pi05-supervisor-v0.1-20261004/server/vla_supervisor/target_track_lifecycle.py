"""Fail-closed lifecycle for a language-selected visual target track.

Semantic localization is allowed only before manipulation starts.  Once a
target has been confirmed, tracking may preserve its identity through a short
occlusion, but neither tracking nor action sensitivity may silently replace it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable

from .semantic_goal_selector import SemanticGoalDecision


@dataclass(frozen=True)
class TargetTrackDecision:
    state: str
    target_candidate_id: Hashable | None
    confidence: float
    actionable: bool
    reason: str
    diagnostics: dict[str, object]


class TargetTrackLifecycle:
    """Lock an early semantic target and preserve identity conservatively."""

    def __init__(self, *, confirmation_frames: int = 2,
                 minimum_tracker_confidence: float = 0.5,
                 maximum_occluded_frames: int = 3) -> None:
        if confirmation_frames < 1 or maximum_occluded_frames < 0:
            raise ValueError("invalid target lifecycle window")
        if not 0 <= minimum_tracker_confidence <= 1:
            raise ValueError("tracker confidence must lie in [0, 1]")
        self.confirmation_frames = confirmation_frames
        self.minimum_tracker_confidence = minimum_tracker_confidence
        self.maximum_occluded_frames = maximum_occluded_frames
        self.reset()

    def reset(self) -> None:
        self.target_id = None
        self.target_confidence = 0.0
        self.pending_id = None
        self.pending_count = 0
        self.occluded_count = 0
        self.manipulation_started = False

    @staticmethod
    def _diagnostics(counterfactual_top_id, counterfactual_stable):
        return {
            "counterfactual_top_id": counterfactual_top_id,
            "counterfactual_stable": counterfactual_stable,
            "counterfactual_role": "diagnostic_only_not_identity_evidence",
        }

    def observe(self, *, semantic: SemanticGoalDecision | None,
                tracker_candidate_id: Hashable | None,
                tracker_confidence: float | None,
                manipulation_started: bool,
                counterfactual_top_id: Hashable | None = None,
                counterfactual_stable: bool | None = None) -> TargetTrackDecision:
        """Update target identity without converting missing vision to failure."""
        if tracker_confidence is not None and not 0 <= tracker_confidence <= 1:
            raise ValueError("tracker confidence must lie in [0, 1]")
        self.manipulation_started = self.manipulation_started or manipulation_started
        diagnostics = self._diagnostics(counterfactual_top_id, counterfactual_stable)

        if self.target_id is None:
            if self.manipulation_started:
                return TargetTrackDecision(
                    "unavailable_after_manipulation", None, 0.0, False,
                    "target was not locked before manipulation; late identity inference is forbidden",
                    diagnostics)
            candidate = (
                semantic.target_candidate_id
                if semantic is not None and semantic.state == "target_candidate"
                else None
            )
            if candidate is None:
                self.pending_id = None
                self.pending_count = 0
                return TargetTrackDecision(
                    "acquiring", None, 0.0, False,
                    "semantic evidence is absent or ambiguous", diagnostics)
            if candidate == self.pending_id:
                self.pending_count += 1
            else:
                self.pending_id = candidate
                self.pending_count = 1
            if self.pending_count < self.confirmation_frames:
                return TargetTrackDecision(
                    "acquiring", None, float(semantic.score), False,
                    "waiting for temporally consistent semantic selection", diagnostics)
            self.target_id = candidate
            self.target_confidence = float(semantic.score)
            return TargetTrackDecision(
                "locked", self.target_id, self.target_confidence, True,
                "target locked before manipulation", diagnostics)

        tracker_valid = (
            tracker_candidate_id is not None
            and tracker_confidence is not None
            and tracker_confidence >= self.minimum_tracker_confidence
        )
        if tracker_valid and tracker_candidate_id == self.target_id:
            self.occluded_count = 0
            self.target_confidence = min(self.target_confidence, float(tracker_confidence))
            return TargetTrackDecision(
                "tracked", self.target_id, self.target_confidence, True,
                "tracker preserves the locked target identity", diagnostics)
        if tracker_valid and tracker_candidate_id != self.target_id:
            return TargetTrackDecision(
                "identity_conflict", self.target_id, 0.0, False,
                "tracker candidate conflicts with the locked target; automatic identity switch is forbidden",
                diagnostics)

        self.occluded_count += 1
        if self.occluded_count <= self.maximum_occluded_frames:
            return TargetTrackDecision(
                "temporarily_occluded", self.target_id, 0.0, False,
                "locked identity retained but unavailable for actionable fusion", diagnostics)
        return TargetTrackDecision(
            "lost", self.target_id, 0.0, False,
            "target track exceeded the occlusion budget", diagnostics)

    @staticmethod
    def semantic_scores(decision: TargetTrackDecision) -> dict[Hashable, float]:
        """Adapter for control-goal fusion; unknown states provide no score."""
        if not decision.actionable or decision.target_candidate_id is None:
            return {}
        return {decision.target_candidate_id: decision.confidence}
