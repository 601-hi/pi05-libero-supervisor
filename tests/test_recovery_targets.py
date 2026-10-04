from vla_supervisor.events import EventType, MonitorEvent
from vla_supervisor.recovery_targets import RecoveryTarget, RecoveryTargetRouter


def event(state, confidence=.9, source="visual_consequence"):
    return MonitorEvent(EventType.OBJECT_FAILURE, 1.0, confidence,
                        evidence={"diagnostic_state": state}, source=source)


def test_routes_wrong_object_to_release_and_reacquire():
    result = RecoveryTargetRouter().route([event("wrong_object_control")])
    assert result.target is RecoveryTarget.RELEASE_WRONG_OBJECT
    assert result.actionable


def test_routes_failed_placement_to_goal_relation_not_full_task_restart():
    result = RecoveryTargetRouter().route([event("goal_relation_not_satisfied")])
    assert result.target is RecoveryTarget.RESTORE_GOAL_RELATION
    assert result.actionable


def test_low_confidence_diagnosis_names_target_but_withholds_actionability():
    result = RecoveryTargetRouter().route([event("empty_grasp_or_miss", confidence=.6)])
    assert result.target is RecoveryTarget.ESTABLISH_STABLE_CONTROL
    assert not result.actionable


def test_policy_stall_alone_requests_evidence():
    stall = MonitorEvent(EventType.POLICY_STALL, 1.0, .95, source="policy_stall")
    result = RecoveryTargetRouter().route([stall])
    assert result.target is RecoveryTarget.REQUEST_MORE_EVIDENCE
    assert not result.actionable


def test_priority_prevents_goal_relation_from_hiding_wrong_object():
    result = RecoveryTargetRouter().route([
        event("goal_relation_not_satisfied", source="goal_progress"),
        event("wrong_object_control", source="semantic_consequence"),
    ])
    assert result.target is RecoveryTarget.RELEASE_WRONG_OBJECT


def test_reliable_progress_states_route_to_the_missing_milestone():
    router = RecoveryTargetRouter()
    assert router.route([event("approach_progress_stalled")]).target is RecoveryTarget.REACQUIRE_INTENDED_TARGET
    assert router.route([event("grasp_progress_stalled")]).target is RecoveryTarget.ESTABLISH_STABLE_CONTROL
    assert router.route([event("goal_relation_progress_stalled")]).target is RecoveryTarget.RESTORE_GOAL_RELATION
