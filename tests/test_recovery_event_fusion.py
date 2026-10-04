from vla_supervisor.recovery_event_fusion import (
    DiagnosticState,
    EpisodeBelief,
    EventEvidence,
    update_belief,
)


def test_incomplete_evidence_abstains():
    belief = update_belief(EpisodeBelief(), EventEvidence("e0", world_motion=False))
    assert belief.state == DiagnosticState.PENDING


def test_empty_first_grasp_is_superseded_by_later_control():
    first = update_belief(EpisodeBelief(), EventEvidence(
        "e0", world_motion=False, wrist_attachment=False, observation_complete=True
    ))
    assert first.state == DiagnosticState.EMPTY_GRASP_OR_MISS
    second = update_belief(first, EventEvidence(
        "e1", world_motion=True, wrist_attachment=True, semantic_target_match=True
    ))
    assert second.state == DiagnosticState.CONTROL_ESTABLISHED
    assert second.established_by_event == "e1"


def test_wrong_object_requires_physical_control_and_semantic_mismatch():
    belief = update_belief(EpisodeBelief(), EventEvidence(
        "e0", world_motion=True, wrist_attachment=True, semantic_target_match=False
    ))
    assert belief.state == DiagnosticState.WRONG_OBJECT_CONTROL


def test_semantics_alone_cannot_establish_control():
    belief = update_belief(EpisodeBelief(), EventEvidence(
        "e0", semantic_target_match=True, observation_complete=True
    ))
    assert belief.state == DiagnosticState.PENDING


def test_goal_relation_only_after_control():
    before = update_belief(EpisodeBelief(), EventEvidence(
        "e0", goal_relation_satisfied=False, observation_complete=True
    ))
    assert before.state == DiagnosticState.PENDING
    controlled = update_belief(before, EventEvidence(
        "e1", world_motion=True, wrist_attachment=True, semantic_target_match=True
    ))
    after = update_belief(controlled, EventEvidence("e2", goal_relation_satisfied=False))
    assert after.state == DiagnosticState.GOAL_RELATION_NOT_SATISFIED


def test_control_persists_until_release_or_loss():
    controlled = update_belief(EpisodeBelief(), EventEvidence(
        "e0", world_motion=True, wrist_attachment=True
    ))
    ambiguous = update_belief(controlled, EventEvidence("e1"))
    assert ambiguous.state == DiagnosticState.CONTROL_ESTABLISHED
    lost = update_belief(ambiguous, EventEvidence("e2", attachment_lost=True))
    assert lost.state == DiagnosticState.CONTROL_LOST
