"""Explicit fusion of causal control and semantic task-target evidence."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Mapping

from .candidate_relation_graph import CandidateRelationGraph


@dataclass(frozen=True)
class ControlGoalDecision:
    state: str
    controlled_candidate_id: Hashable | None
    target_candidate_id: Hashable | None
    semantic_margin: float
    reason: str


def fuse_control_and_goal(
    *,
    controlled_candidate_id: Hashable | None,
    semantic_scores: Mapping[Hashable, float],
    relation_graph: CandidateRelationGraph,
    minimum_semantic_score: float = 0.5,
    minimum_semantic_margin: float = 0.1,
) -> ControlGoalDecision:
    """Keep physical control and language consistency separately observable.

    ``semantic_scores`` are calibrated evidence, not assumed probabilities.
    They may come from a detector, a spatial-language module, or annotations.
    Part-to-whole lifting is permitted only when the larger candidate is also
    semantically supported; geometry alone cannot invent object identity.
    """
    ranked = sorted(semantic_scores.items(), key=lambda item: item[1], reverse=True)
    target_id = ranked[0][0] if ranked else None
    best = float(ranked[0][1]) if ranked else float("-inf")
    runner_up = float(ranked[1][1]) if len(ranked) > 1 else float("-inf")
    margin = float("inf") if len(ranked) == 1 else best - runner_up
    if controlled_candidate_id is None:
        return ControlGoalDecision("no_control", None, target_id, margin, "causal selector has not established control")
    if target_id is None or best < minimum_semantic_score or margin < minimum_semantic_margin:
        return ControlGoalDecision("ambiguous", controlled_candidate_id, target_id, margin, "semantic target evidence is insufficient")

    compatible = set(relation_graph.equivalents(controlled_candidate_id))
    compatible.add(controlled_candidate_id)
    for whole_id in relation_graph.whole_candidates_for(controlled_candidate_id):
        if semantic_scores.get(whole_id, float("-inf")) >= minimum_semantic_score:
            compatible.add(whole_id)
    if target_id in compatible:
        return ControlGoalDecision("target_controlled", controlled_candidate_id, target_id, margin, "causal and semantic evidence agree")
    return ControlGoalDecision("wrong_object_control", controlled_candidate_id, target_id, margin, "controlled candidate conflicts with semantic target")

