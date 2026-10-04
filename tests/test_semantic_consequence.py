from vla_supervisor.goal_relations import parse_goal_relation
from vla_supervisor.events import EventType, RecommendedAction
from vla_supervisor.semantic_consequence import (
    CalibratedSemanticConsequenceAdapter,
    SemanticConsequenceDecision,
    SemanticConsequenceEvidence,
    SemanticConsequenceGate,
    SemanticConsequenceMonitor,
)


def test_goal_parser_supports_open_and_binary_switches():
    assert parse_goal_relation("open the middle drawer of the cabinet").relation == "open"
    assert parse_goal_relation("turn off the stove").relation == "turn_off"
    assert parse_goal_relation("turn on the stove").relation == "turn_on"


def test_semantic_gate_abstains_without_observability_or_calibration():
    gate = SemanticConsequenceGate()
    assert gate.decide(SemanticConsequenceEvidence(False, True, goal_satisfied_score=.9)).state == "unknown"
    assert gate.decide(SemanticConsequenceEvidence(True, False, wrong_object_score=.9)).state == "unknown"


def test_semantic_gate_rejects_conflicting_hypotheses():
    decision = SemanticConsequenceGate().decide(SemanticConsequenceEvidence(
        True, True, goal_satisfied_score=.81, wrong_object_score=.76,
    ))
    assert decision.state == "ambiguous"


def test_semantic_gate_emits_typed_strong_evidence():
    decision = SemanticConsequenceGate().decide(SemanticConsequenceEvidence(
        True, True, goal_satisfied_score=.1, object_lost_score=.9,
    ))
    assert decision.state == "object_lost"


def test_no_progress_requires_temporal_confirmation():
    monitor = SemanticConsequenceMonitor()
    decision = SemanticConsequenceDecision("no_progress", .9, .8, "test")
    first = monitor.observe_decision(decision, action_index=10)
    second = monitor.observe_decision(decision, action_index=11)
    assert first.event_type is EventType.OBJECT_AMBIGUOUS
    assert first.recommended_action is RecommendedAction.REQUEST_MORE_EVIDENCE
    assert second.event_type is EventType.OBJECT_FAILURE
    assert second.recommended_action is RecommendedAction.STOP_AND_REPLAN


def test_wrong_object_is_immediate_but_unknown_abstains():
    monitor = SemanticConsequenceMonitor()
    unknown = monitor.observe_decision(
        SemanticConsequenceDecision("unknown", 0, 0, "not visible"), action_index=2
    )
    wrong = monitor.observe_decision(
        SemanticConsequenceDecision("wrong_object", .9, .8, "conflict"), action_index=3
    )
    assert unknown.event_type is EventType.OBJECT_AMBIGUOUS
    assert wrong.event_type is EventType.OBJECT_FAILURE
    assert wrong.evidence["diagnostic_state"] == "wrong_object_control"


def test_calibrated_no_progress_routes_by_reliable_phase_only_after_confirmation():
    monitor = SemanticConsequenceMonitor()
    decision = SemanticConsequenceGate().decide(SemanticConsequenceEvidence(
        True, True, no_progress_score=.9,
        metadata={"phase": "place", "phase_reliable": True},
    ))
    first = monitor.observe_decision(decision, action_index=10)
    second = monitor.observe_decision(decision, action_index=20)
    assert first.event_type is EventType.OBJECT_AMBIGUOUS
    assert second.evidence["diagnostic_state"] == "goal_relation_progress_stalled"


def test_unreliable_phase_does_not_invent_milestone():
    decision = SemanticConsequenceDecision(
        "no_progress", .9, .8, "test", {"phase": "grasp", "phase_reliable": False}
    )
    event = SemanticConsequenceMonitor().observe_decision(decision, action_index=1)
    assert "diagnostic_state" not in event.evidence


def test_online_adapter_firewalls_history_and_shadows_failure_by_default():
    captured = {}

    def scorer(_before, _after, _action, history):
        captured["history"] = history
        return SemanticConsequenceEvidence(
            True, True, wrong_object_score=.95,
            metadata={"phase": "grasp", "phase_reliable": True},
        )

    adapter = CalibratedSemanticConsequenceAdapter(scorer)
    event = adapter.observe_conditioned(
        upstream_events=(), images_before={}, images_after={},
        intended_action=[0] * 7,
        history=({"action_index": 1, "intended_action": [0] * 7,
                  "state_after": {"eef_pos": [1, 2, 3]}, "reward": 1},),
        action_index=2,
    )
    assert captured["history"] == ({"action_index": 1, "intended_action": [0] * 7},)
    assert event.event_type is EventType.OBJECT_AMBIGUOUS
    assert event.evidence["shadow_original_event_type"] == "object_failure"
    assert event.evidence["diagnostic_state"] == "wrong_object_control"


def test_online_adapter_never_promotes_uncalibrated_evidence():
    adapter = CalibratedSemanticConsequenceAdapter(
        lambda *_: SemanticConsequenceEvidence(True, False, no_progress_score=.99),
        control_enabled=True,
    )
    event = adapter.observe_conditioned(
        upstream_events=(), images_before={}, images_after={},
        intended_action=[0] * 7, history=(), action_index=2,
    )
    assert event.event_type is EventType.NORMAL
    assert event.evidence["semantic_state"] == "unknown"
