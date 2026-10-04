"""Conservative event-level fusion for recovery diagnosis.

This module deliberately emits diagnostic states, not automatic robot actions.
Unknown or conflicting evidence is represented explicitly by ``pending``.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum


class DiagnosticState(str, Enum):
    PENDING = "pending"
    EMPTY_GRASP_OR_MISS = "empty_grasp_or_miss"
    CONTROL_ESTABLISHED = "control_established"
    WRONG_OBJECT_CONTROL = "wrong_object_control"
    CONTROL_LOST = "control_lost"
    GOAL_RELATION_NOT_SATISFIED = "goal_relation_not_satisfied"
    GOAL_RELATION_SATISFIED = "goal_relation_satisfied"


@dataclass(frozen=True)
class EventEvidence:
    event_id: str
    world_motion: bool | None = None
    wrist_attachment: bool | None = None
    semantic_target_match: bool | None = None
    release_observed: bool = False
    attachment_lost: bool = False
    goal_relation_satisfied: bool | None = None
    observation_complete: bool = False


@dataclass(frozen=True)
class EpisodeBelief:
    state: DiagnosticState = DiagnosticState.PENDING
    controlled_object_known_target: bool | None = None
    established_by_event: str | None = None
    last_event_id: str | None = None


def update_belief(prior: EpisodeBelief, evidence: EventEvidence) -> EpisodeBelief:
    """Update the episode belief without turning missing evidence into certainty."""
    if evidence.release_observed or evidence.attachment_lost:
        return EpisodeBelief(
            state=DiagnosticState.CONTROL_LOST,
            controlled_object_known_target=None,
            established_by_event=prior.established_by_event,
            last_event_id=evidence.event_id,
        )

    physical_control = evidence.world_motion is True and evidence.wrist_attachment is True
    if physical_control:
        if evidence.semantic_target_match is False:
            state = DiagnosticState.WRONG_OBJECT_CONTROL
            target = False
        else:
            state = DiagnosticState.CONTROL_ESTABLISHED
            target = True if evidence.semantic_target_match is True else None
        belief = EpisodeBelief(state, target, evidence.event_id, evidence.event_id)
    elif (
        evidence.observation_complete
        and evidence.world_motion is False
        and evidence.wrist_attachment is False
    ):
        # A later close can replace this hypothesis; it is not terminal.
        belief = EpisodeBelief(
            DiagnosticState.EMPTY_GRASP_OR_MISS, None, prior.established_by_event, evidence.event_id
        )
    elif prior.state in {
        DiagnosticState.CONTROL_ESTABLISHED,
        DiagnosticState.WRONG_OBJECT_CONTROL,
    }:
        belief = replace(prior, last_event_id=evidence.event_id)
    else:
        belief = EpisodeBelief(
            DiagnosticState.PENDING,
            prior.controlled_object_known_target,
            prior.established_by_event,
            evidence.event_id,
        )

    # Goal relation is meaningful only after physical control was established.
    control_was_established = belief.established_by_event is not None
    if control_was_established and evidence.goal_relation_satisfied is not None:
        return replace(
            belief,
            state=(
                DiagnosticState.GOAL_RELATION_SATISFIED
                if evidence.goal_relation_satisfied
                else DiagnosticState.GOAL_RELATION_NOT_SATISFIED
            ),
        )
    return belief
