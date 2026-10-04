from vla_supervisor.contact_diagnosis import ContactDiagnosticObjectMonitor
from vla_supervisor.diagnostic_router import AmbiguityGatedObjectMonitor, ShadowObjectMonitor
from vla_supervisor.events import EventType, MonitorEvent, RecommendedAction


def upstream(decision="unknown", reliability=.2):
    return [MonitorEvent(
        EventType.EXECUTION_AMBIGUOUS,
        0.0,
        reliability,
        {"decision": decision, "reliability": reliability},
        RecommendedAction.REQUEST_MORE_EVIDENCE,
        "four_state_execution_consistency",
        0,
    )]


class SequenceEvidence:
    def __init__(self, sequence):
        self.sequence = iter(sequence)

    def reset(self):
        pass

    def __call__(self, images_before, images_after, intended_action, history):
        # The adapter must strip robot states, reward, and result labels.
        assert all(set(item) <= {"action_index", "intended_action"} for item in history)
        return next(self.sequence)


def visual(**updates):
    values = dict(
        manipulation_phase=True,
        gripper_closed=True,
        background_confidence=.9,
        gripper_candidate_proximity=.9,
        independent_motion_probability=None,
        attachment_probability=None,
        blocked_motion_probability=None,
        semantic_target_probability=None,
    )
    values.update(updates)
    return values


def invoke(router, index):
    return router.observe_conditioned(
        upstream_events=upstream(),
        images_before={},
        images_after={},
        intended_action=[0, 0, 0, 0, 0, 0, 1],
        history=[{"action_index": index, "intended_action": [0] * 7,
                  "state_after": {"forbidden": True}, "success": True}],
        action_index=index,
    )


def test_selective_router_passes_execution_context_into_contact_state_machine():
    scorer = SequenceEvidence([
        visual(), visual(),
        visual(independent_motion_probability=.9, attachment_probability=.9),
        visual(independent_motion_probability=.9, attachment_probability=.9),
        visual(independent_motion_probability=.9, attachment_probability=.9),
        visual(semantic_target_probability=.95, attachment_probability=.9),
    ])
    router = AmbiguityGatedObjectMonitor(ContactDiagnosticObjectMonitor(scorer))
    outputs = [invoke(router, index) for index in range(6)]
    assert outputs[1].evidence["diagnostic_state"] == "contact_suspected"
    assert outputs[4].evidence["diagnostic_state"] == "object_control_established"
    assert outputs[5].evidence["diagnostic_state"] == "target_controlled"
    assert outputs[5].event_type is EventType.NORMAL


def test_fixed_obstacle_path_becomes_actionable_after_causal_observation():
    scorer = SequenceEvidence([
        visual(), visual(),
        visual(independent_motion_probability=.1, blocked_motion_probability=.9),
        visual(independent_motion_probability=.1, blocked_motion_probability=.9),
        visual(independent_motion_probability=.1, blocked_motion_probability=.9),
    ])
    router = AmbiguityGatedObjectMonitor(ContactDiagnosticObjectMonitor(scorer))
    outputs = [invoke(router, index) for index in range(5)]
    assert outputs[-1].evidence["diagnostic_state"] == "fixed_obstacle_or_jam"
    assert outputs[-1].event_type is EventType.OBJECT_FAILURE
    assert outputs[-1].recommended_action is RecommendedAction.STOP_AND_REPLAN


def test_shadow_monitor_preserves_diagnosis_but_cannot_request_control():
    scorer = SequenceEvidence([
        visual(), visual(),
        visual(independent_motion_probability=.1, blocked_motion_probability=.9),
        visual(independent_motion_probability=.1, blocked_motion_probability=.9),
        visual(independent_motion_probability=.1, blocked_motion_probability=.9),
    ])
    monitor = ShadowObjectMonitor(ContactDiagnosticObjectMonitor(scorer))
    router = AmbiguityGatedObjectMonitor(monitor)
    outputs = [invoke(router, index) for index in range(5)]
    final = outputs[-1]
    assert final.evidence["diagnostic_state"] == "fixed_obstacle_or_jam"
    assert final.evidence["shadow_original_event_type"] == "object_failure"
    assert final.event_type is EventType.OBJECT_AMBIGUOUS
    assert final.recommended_action is RecommendedAction.REQUEST_MORE_EVIDENCE
