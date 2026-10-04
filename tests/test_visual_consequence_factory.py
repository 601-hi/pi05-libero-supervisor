from vla_supervisor.events import EventType, MonitorEvent
from vla_supervisor.contact_diagnosis import ContactDiagnosticObjectMonitor
from vla_supervisor.visual_consequence import BackgroundConsequenceEvidenceScorer


def test_background_only_contact_adapter_logs_abstention_diagnostics():
    scorer = BackgroundConsequenceEvidenceScorer()
    monitor = ContactDiagnosticObjectMonitor(scorer)
    upstream = [MonitorEvent(
        EventType.EXECUTION_AMBIGUOUS, 1.0, .2,
        evidence={"decision": "unknown", "reliability": .2},
    )]
    event = monitor.observe_with_execution_context(
        upstream_events=upstream,
        images_before=None,
        images_after=None,
        intended_action=[0] * 7,
        history=(),
        action_index=1,
    )
    assert event.event_type is EventType.NORMAL
    assert event.evidence["diagnostic_state"] == "free_motion"
    assert event.evidence["visual_diagnostics"]["status"] == "missing_paired_images"
