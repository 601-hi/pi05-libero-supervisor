"""Conservative semantic target selection from externally supplied evidence."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Mapping


@dataclass(frozen=True)
class SemanticCandidateEvidence:
    category_score: float | None
    source_relation_score: float | None
    detector_confidence: float
    observable: bool = True


@dataclass(frozen=True)
class SemanticGoalDecision:
    target_candidate_id: Hashable | None
    state: str
    score: float
    margin: float
    candidate_scores: Mapping[Hashable, float]


class SemanticGoalSelector:
    """Require both object-category and source-relation support.

    Scores are treated as bounded evidence, not Bayesian probabilities.  A
    geometric mean prevents a high category score from hiding a failed source
    relation (for example, selecting the right kind of bowl in the wrong
    drawer).  Unsupported or uncalibrated upstream models should pass ``None``
    and receive an honest rejection.
    """

    def __init__(
        self,
        *,
        minimum_category_score: float = 0.5,
        minimum_relation_score: float = 0.5,
        minimum_detector_confidence: float = 0.3,
        minimum_joint_score: float = 0.55,
        minimum_margin: float = 0.1,
    ) -> None:
        for value in (
            minimum_category_score,
            minimum_relation_score,
            minimum_detector_confidence,
            minimum_joint_score,
            minimum_margin,
        ):
            if not 0 <= value <= 1:
                raise ValueError("semantic thresholds must lie in [0, 1]")
        self.minimum_category = minimum_category_score
        self.minimum_relation = minimum_relation_score
        self.minimum_detector = minimum_detector_confidence
        self.minimum_joint = minimum_joint_score
        self.minimum_margin = minimum_margin

    def select(
        self,
        evidence: Mapping[Hashable, SemanticCandidateEvidence],
        *,
        relation_required: bool,
    ) -> SemanticGoalDecision:
        scores: dict[Hashable, float] = {}
        for candidate_id, item in evidence.items():
            if not item.observable or item.detector_confidence < self.minimum_detector:
                continue
            if item.category_score is None or not 0 <= item.category_score <= 1:
                continue
            if item.category_score < self.minimum_category:
                continue
            if relation_required:
                if item.source_relation_score is None or not 0 <= item.source_relation_score <= 1:
                    continue
                if item.source_relation_score < self.minimum_relation:
                    continue
                joint = (item.category_score * item.source_relation_score) ** 0.5
            else:
                joint = item.category_score
            scores[candidate_id] = float(joint * item.detector_confidence)
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        if not ranked:
            return SemanticGoalDecision(None, "unobservable_or_unsupported", 0.0, 0.0, scores)
        best_id, best = ranked[0]
        runner_up = ranked[1][1] if len(ranked) > 1 else 0.0
        margin = best - runner_up
        if best < self.minimum_joint:
            return SemanticGoalDecision(None, "low_confidence", best, margin, scores)
        if margin < self.minimum_margin:
            return SemanticGoalDecision(None, "ambiguous", best, margin, scores)
        return SemanticGoalDecision(best_id, "target_candidate", best, margin, scores)

