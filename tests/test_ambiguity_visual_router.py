import numpy as np

from vla_supervisor.diagnostic_router import (AmbiguityGatedObjectMonitor,
                                              GripperInteractionGate,
                                              VisualRoutingConfig)
from vla_supervisor.events import EventType, MonitorEvent, RecommendedAction


class CountingVisual:
    def __init__(self, event_type=EventType.NORMAL):
        self.calls = 0
        self.event_type = event_type

    def reset(self):
        self.calls = 0

    def observe(self, **kwargs):
        self.calls += 1
        action = (RecommendedAction.STOP_AND_REPLAN
                  if self.event_type is EventType.OBJECT_FAILURE else RecommendedAction.CONTINUE)
        return MonitorEvent(self.event_type, float(self.event_type is not EventType.NORMAL), .9,
                            {"mechanism": "test"}, action, "counting_visual", kwargs["action_index"])


def event(kind, confidence=.9):
    return MonitorEvent(kind, 1.0, confidence, {}, RecommendedAction.CONTINUE, "execution", 1)


def observe(router, upstream):
    return router.observe_conditioned(upstream_events=upstream, images_before={}, images_after={},
                                      intended_action=np.zeros(7), history=(), action_index=1)


def test_normal_execution_does_not_invoke_visual():
    visual = CountingVisual()
    router = AmbiguityGatedObjectMonitor(visual)
    result = observe(router, [event(EventType.NORMAL)])
    assert visual.calls == 0
    assert result.evidence == {"visual_invoked": False, "routing_reason": "execution_not_ambiguous"}


def test_ambiguity_invokes_visual_and_preserves_structured_result():
    visual = CountingVisual(EventType.OBJECT_FAILURE)
    router = AmbiguityGatedObjectMonitor(visual)
    result = observe(router, [event(EventType.EXECUTION_AMBIGUOUS)])
    assert visual.calls == 1
    assert result.event_type is EventType.OBJECT_FAILURE
    assert result.evidence["visual_invoked"] is True
    assert result.evidence["routing_reason"] == "execution_ambiguous"


def test_clear_execution_alarm_cannot_be_vetoed_by_visual():
    visual = CountingVisual()
    router = AmbiguityGatedObjectMonitor(visual)
    result = observe(router, [event(EventType.EXECUTION_MISMATCH)])
    assert visual.calls == 0
    assert result.evidence["routing_reason"] == "clear_upstream_decision"


def test_followup_window_keeps_visual_alive_briefly():
    visual = CountingVisual()
    router = AmbiguityGatedObjectMonitor(visual, VisualRoutingConfig(keep_alive_steps=2))
    observe(router, [event(EventType.EXECUTION_AMBIGUOUS)])
    one = observe(router, [event(EventType.NORMAL)])
    two = observe(router, [event(EventType.NORMAL)])
    three = observe(router, [event(EventType.NORMAL)])
    assert one.evidence["routing_reason"] == "ambiguity_followup"
    assert two.evidence["routing_reason"] == "ambiguity_followup"
    assert three.evidence["visual_invoked"] is False


def test_gripper_phase_gate_only_opens_after_sign_transition():
    gate = GripperInteractionGate(active_steps_after_transition=3)
    open_action = np.asarray([0, 0, 0, 0, 0, 0, -1.0])
    close_action = np.asarray([0, 0, 0, 0, 0, 0, 1.0])
    assert gate.update(open_action) is False
    assert gate.update(open_action) is False
    assert gate.update(close_action) is True
    assert gate.update(close_action) is True
    assert gate.update(close_action) is True
    assert gate.update(close_action) is False


def test_phase_gate_blocks_ambiguous_visual_outside_interaction():
    visual = CountingVisual()
    router = AmbiguityGatedObjectMonitor(
        visual, phase_gate=GripperInteractionGate(active_steps_after_transition=2))
    result = router.observe_conditioned(
        upstream_events=[event(EventType.EXECUTION_AMBIGUOUS)], images_before={}, images_after={},
        intended_action=np.asarray([0, 0, 0, 0, 0, 0, -1.0]), history=(), action_index=1)
    assert visual.calls == 0
    assert result.evidence["routing_reason"] == "outside_manipulation_phase"
