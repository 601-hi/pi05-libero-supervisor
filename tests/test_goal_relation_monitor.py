from vla_supervisor.goal_relation_monitor import (
    GoalEvent,
    GoalRelationMonitor,
    RelationEvidence,
    TruthState,
)


def ev(goal, state, progress, t, confidence=0.9):
    return RelationEvidence(goal, "Close", state, progress, confidence, t, provenance="test")


def test_completion_requires_temporal_confirmation():
    m = GoalRelationMonitor(("drawer_closed",), confirmation_frames=3)
    assert m.update([ev("drawer_closed", TruthState.SATISFIED, 1.0, 0)]) == GoalEvent.CONTINUE
    assert m.update([ev("drawer_closed", TruthState.SATISFIED, 1.0, 1)]) == GoalEvent.CONTINUE
    assert m.update([ev("drawer_closed", TruthState.SATISFIED, 1.0, 2)]) == GoalEvent.COMPLETE


def test_stalled_goal_is_not_execution_failure():
    m = GoalRelationMonitor(("drawer_closed",), no_progress_window=5, min_progress_gain=0.03)
    result = GoalEvent.CONTINUE
    for t, p in enumerate([0.20, 0.21, 0.20, 0.21, 0.20]):
        result = m.update([ev("drawer_closed", TruthState.VIOLATED, p, t)])
    assert result == GoalEvent.GOAL_NO_PROGRESS


def test_unknown_requests_reobservation_instead_of_false_alarm():
    m = GoalRelationMonitor(("stove_on",), unknown_patience=3)
    result = GoalEvent.CONTINUE
    for t in range(3):
        result = m.update([ev("stove_on", TruthState.UNOBSERVABLE, None, t, 0.1)])
    assert result == GoalEvent.NEED_REOBSERVATION


def test_confirmed_goal_can_regress():
    m = GoalRelationMonitor(("bowl_on_plate",), confirmation_frames=2)
    m.update([ev("bowl_on_plate", TruthState.SATISFIED, 1.0, 0)])
    assert m.update([ev("bowl_on_plate", TruthState.SATISFIED, 1.0, 1)]) == GoalEvent.COMPLETE
    assert m.update([ev("bowl_on_plate", TruthState.VIOLATED, 0.1, 2)]) == GoalEvent.GOAL_REGRESSION


def test_composite_goal_requires_every_atom():
    m = GoalRelationMonitor(("mug_inside", "microwave_closed"), confirmation_frames=2)
    for t in range(2):
        result = m.update([
            ev("mug_inside", TruthState.SATISFIED, 1.0, t),
            ev("microwave_closed", TruthState.VIOLATED, 0.4, t),
        ])
    assert result != GoalEvent.COMPLETE
