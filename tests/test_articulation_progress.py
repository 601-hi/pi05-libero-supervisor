from vla_supervisor.articulation_progress import (
    ArticulationMeasurement,
    ArticulationProgressConfig,
    ArticulationProgressMonitor,
    ArticulationProgressWatchdog,
)
from vla_supervisor.events import EventType
from vla_supervisor.fusion import PersistenceRule, TemporalFusion
from vla_supervisor.instruction_guard import GuardConfig, InstructionGuard
from vla_supervisor.monitors import NullObjectResultMonitor
from vla_supervisor.recovery import RecoveryController
from vla_supervisor.runtime import SupervisorRuntime


CFG = ArticulationProgressConfig(
    "close-v1", initial_grace_actions=8, plateau_window_actions=3,
    regression_confirmations=2, smoothing_samples=1)


def measurement(index, area, **overrides):
    values = dict(action_index=index, area_fraction=area, confidence=.9,
                  observable=True, identity_reliable=True, calibrated=True,
                  calibration_id="close-v1", relation_id="drawer_closed",
                  predicate="Close", provenance="test")
    values.update(overrides)
    return ArticulationMeasurement(**values)


def test_progress_then_plateau_latches_no_progress():
    watchdog = ArticulationProgressWatchdog(CFG)
    states = [watchdog.update(measurement(i, area)).state
              for i, area in enumerate([1, .8, .7, .7, .7, .7])]
    assert states[-1] == "goal_no_progress"
    assert watchdog.update(measurement(6, .6)).state == "goal_no_progress"


def test_regression_is_distinct_from_stall():
    watchdog = ArticulationProgressWatchdog(CFG)
    states = [watchdog.update(measurement(i, area)).state
              for i, area in enumerate([1, .5, .5, .9, .9])]
    assert states[-1] == "goal_regression"


def test_unobservable_or_uncalibrated_measurement_abstains():
    watchdog = ArticulationProgressWatchdog(CFG)
    assert watchdog.update(measurement(0, 1, observable=False)).state == "unknown"
    assert watchdog.update(measurement(0, 1, calibrated=False)).state == "unknown"


def test_shadow_and_control_authority_are_explicit():
    class Provider:
        def __init__(self): self.index = 0
        def __call__(self, *_):
            value = measurement(self.index, [1, .8, .7, .7, .7, .7][self.index])
            self.index += 1
            return value
    shadow = ArticulationProgressMonitor(Provider(), CFG, control_enabled=False)
    control = ArticulationProgressMonitor(Provider(), CFG, control_enabled=True)
    for index in range(6):
        kwargs = dict(upstream_events=(), images_before={}, images_after={},
                      intended_action=[0] * 7, history=(), action_index=index)
        shadow_event = shadow.observe_conditioned(**kwargs)
        control_event = control.observe_conditioned(**kwargs)
    assert shadow_event.event_type is EventType.OBJECT_AMBIGUOUS
    assert shadow_event.evidence["shadow_original_event_type"] == "object_failure"
    assert control_event.event_type is EventType.OBJECT_FAILURE
    assert control_event.evidence["diagnostic_state"] == "goal_relation_progress_stalled"


class SequenceProvider:
    def __init__(self, areas): self.areas = areas
    def reset(self): pass
    def __call__(self, _before, _after, _action, _history, action_index):
        return measurement(action_index, self.areas[action_index])


def make_runtime(*, control_enabled):
    relation = ArticulationProgressMonitor(
        SequenceProvider([1, .8, .7, .7, .7, .7]), CFG,
        control_enabled=control_enabled)
    return SupervisorRuntime(
        InstructionGuard(GuardConfig()), [], NullObjectResultMonitor(),
        TemporalFusion({EventType.OBJECT_FAILURE: PersistenceRule(1, 1, .5)}),
        RecoveryController(), conditioned_monitors=(relation,),
        control_enabled=control_enabled)


def robot_state():
    return {"eef_pos": [0, 0, 0], "joint_pos": [0] * 7, "joint_vel": [0] * 7}


def test_runtime_routes_relation_stall_to_hold_and_restore_goal_relation():
    runtime = make_runtime(control_enabled=True)
    runtime.install_chunk([[0] * 7] * 8)
    for index in range(6):
        action, _ = runtime.next_action(robot_state(), index)
        decision = runtime.observe_step(
            intended_action=action, state_before=robot_state(), state_after=robot_state(),
            action_index=index, images_before={}, images_after={})
    assert decision.request_replan
    assert not runtime.chunk
    assert runtime.last_intervention.mode.value == "hold_then_replan"
    assert runtime.last_intervention.recovery_target == "restore_goal_relation"
    assert "restore the required goal relation" in runtime.last_intervention.recovery_prompt_suffix


def test_runtime_shadow_relation_stall_preserves_policy_queue():
    runtime = make_runtime(control_enabled=False)
    runtime.install_chunk([[0] * 7] * 8)
    for index in range(6):
        action, _ = runtime.next_action(robot_state(), index)
        decision = runtime.observe_step(
            intended_action=action, state_before=robot_state(), state_after=robot_state(),
            action_index=index, images_before={}, images_after={})
    assert not decision.request_replan
    assert len(runtime.chunk) == 2
