import pytest

from vla_supervisor.goal_progress import (
    GoalProgressConfig, GoalProgressMonitor, GoalProgressObservation,
    GoalProgressQuerySchedule,
)
from vla_supervisor.events import RecommendedAction
from vla_supervisor.goal_relations import parse_goal_relation


def config(**changes):
    values = dict(calibration_id='synthetic-test-only', measurement_domain='fixed-v1-scale1',
        metric_kind='image_distance', window_actions=4, max_observation_gap=2,
        minimum_samples=5, minimum_path_m=.01, minimum_progress=.05,
        away_threshold=.1, goal_tolerance=.02, maximum_error_bound=.05)
    values.update(changes)
    return GoalProgressConfig(**values)


def obs(i, error=1., **changes):
    values = dict(action_index=i, phase='approach', target_id='object-a',
        measurement_domain='fixed-v1-scale1', metric_kind='image_distance',
        goal_error=error, error_bound=.001, eef_position=(i*.01,0,0),
        observable=True, identity_reliable=True, phase_reliable=True,
        domain_verified=True, geometry_compensated=True)
    values.update(changes)
    return GoalProgressObservation(**values)


def test_stall_confirmation_uses_disjoint_intervals_not_repeated_windows():
    monitor = GoalProgressMonitor(config())
    results = [monitor.observe(obs(i)) for i in range(9)]
    assert results[4].state == 'stall_candidate' and not results[4].confirmed
    assert all(not r.confirmed for r in results[5:8])
    assert results[8].confirmed and results[8].confirmations == 2
    assert results[8].shadow_event().recommended_action == RecommendedAction.REQUEST_MORE_EVIDENCE


@pytest.mark.parametrize('change', [dict(observable=False), dict(identity_reliable=False),
    dict(phase_reliable=False), dict(domain_verified=False), dict(geometry_compensated=False),
    dict(error_bound=float('nan')), dict(goal_error=float('inf')),
    dict(measurement_domain='other-camera'), dict(error_bound=.2)])
def test_missing_uncertain_or_changed_domain_abstains_and_breaks_confirmation(change):
    monitor = GoalProgressMonitor(config())
    for i in range(5):
        monitor.observe(obs(i))
    assert monitor.observe(obs(5, **change)).state == 'unknown'
    assert monitor.observe(obs(6)).state == 'warming'


def test_approach_and_away_have_opposite_signed_evidence():
    approach, away = GoalProgressMonitor(config()), GoalProgressMonitor(config())
    for i in range(5):
        a = approach.observe(obs(i, 1-i*.1))
        b = away.observe(obs(i, 1+i*.1))
    assert a.state == 'progress'
    assert b.state == 'away_candidate'


def test_phase_target_and_sampling_gap_reset_the_window():
    for changes in (dict(phase='transport'), dict(target_id='destination')):
        monitor = GoalProgressMonitor(config())
        for i in range(4):
            monitor.observe(obs(i))
        assert monitor.observe(obs(4, **changes)).state == 'warming'
    monitor = GoalProgressMonitor(config())
    monitor.observe(obs(0))
    assert monitor.observe(obs(10)).state == 'warming'


def test_contact_proximity_cannot_prove_manipulation_progress():
    monitor = GoalProgressMonitor(config())
    assert monitor.observe(obs(0, phase='manipulate')).state == 'unknown'
    monitor = GoalProgressMonitor(config(metric_kind='task_state_error'))
    assert monitor.observe(obs(0, phase='manipulate', metric_kind='task_state_error')).state == 'warming'


def test_no_calibration_and_goal_band_do_not_trigger_robot_commands():
    assert GoalProgressMonitor().observe(obs(0)).state == 'unknown'
    result = GoalProgressMonitor(config()).observe(obs(0, .005))
    assert result.state == 'within_goal_band'
    assert result.shadow_event().recommended_action == RecommendedAction.REQUEST_MORE_EVIDENCE


def test_stationary_hand_does_not_imply_wrong_instruction():
    monitor = GoalProgressMonitor(config())
    for i in range(5):
        result = monitor.observe(obs(i, eef_position=(0,0,0)))
    assert result.state == 'insufficient_motion'


def test_uncertainty_prevents_false_progress_and_stall():
    monitor = GoalProgressMonitor(config())
    for i in range(5):
        result = monitor.observe(obs(i, 1-i*.01, error_bound=.04))
    assert result.state == 'unknown'


def test_duplicate_or_backward_indices_are_not_extra_confirmation():
    monitor = GoalProgressMonitor(config())
    monitor.observe(obs(3))
    with pytest.raises(ValueError):
        monitor.observe(obs(3))
    monitor.reset()
    assert monitor.observe(obs(0)).state == 'warming'


def test_visual_queries_occur_with_normal_execution_and_are_rate_limited():
    scheduler = GoalProgressQuerySchedule(period_actions=10, minimum_gap=3)
    assert [i for i in range(31) if scheduler.due(i)] == [0,10,20,30]
    scheduler.reset()
    assert [i for i in range(10) if scheduler.due(i, execution_ambiguous=True)] == [0,3,6,9]


def test_prefix_decisions_do_not_depend_on_future_observations():
    prefix = [obs(i) for i in range(9)]
    first = GoalProgressMonitor(config())
    baseline = [first.observe(x) for x in prefix]
    second = GoalProgressMonitor(config())
    extended = [second.observe(x) for x in prefix+[obs(9,.01)]]
    assert baseline == extended[:len(prefix)]


def test_reject_invalid_configuration():
    with pytest.raises(ValueError):
        config(minimum_progress=0)
    with pytest.raises(ValueError):
        config(away_threshold=float('nan'))


@pytest.mark.parametrize('language,object_name,target', [
    ('pick up the alphabet soup and place it in the basket', 'the alphabet soup', 'the basket'),
    ('put the bowl on top of the cabinet', 'the bowl', 'the cabinet'),
    ('pick up the black bowl from table center and place it on the plate', 'the black bowl', 'the plate'),
])
def test_goal_parsing_preserves_manipulated_object_and_destination(language, object_name, target):
    relation = parse_goal_relation(language)
    assert relation.manipulated_object == object_name and relation.target == target
