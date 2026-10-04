from vla_supervisor.events import EventType
from vla_supervisor.monitors import DeterministicTestMonitor


def test_injection_has_explicit_test_evidence_and_exact_interval():
    monitor=DeterministicTestMonitor(EventType.EXECUTION_MISMATCH,4,2)
    kwargs=dict(intended_action=[0]*7,state_before={},state_after={},history=())
    assert monitor.observe(**kwargs,action_index=3).event_type is EventType.NORMAL
    event=monitor.observe(**kwargs,action_index=4)
    assert event.event_type is EventType.EXECUTION_MISMATCH
    assert event.source=="test_injection" and event.evidence["test_only"]
    assert monitor.observe(**kwargs,action_index=6).event_type is EventType.NORMAL
