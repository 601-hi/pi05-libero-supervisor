import numpy as np

from vla_supervisor.checkpoint_recovery import (
    CartesianRollbackController,
    CheckpointBuffer,
    CheckpointSelector,
    FailedPlanNoveltyGate,
    HistoricalCorridorPlanner,
    RecoveryCheckpoint,
    RecoveryEscalationConfig,
    RecoveryEscalationController,
    RecoverySafetyGate,
    WindowedRecoveryResponseMonitor,
)


def checkpoint(index, phase="approach", **kwargs):
    return RecoveryCheckpoint(
        index, (index / 100.0, 0.0, 0.3), tuple([index / 100.0] * 7), phase,
        kwargs.get("observability", .9), kwargs.get("reliability", .9),
        kwargs.get("contact_clear", True), kwargs.get("joint_margin", .2),
        kwargs.get("singularity_margin", .2), kwargs.get("task_progress", .2),
    )


def test_checkpoint_buffer_rejects_untrusted_or_contact_states():
    buffer = CheckpointBuffer()
    assert buffer.consider(checkpoint(1))
    assert not buffer.consider(checkpoint(2))  # spacing gate
    assert not buffer.consider(checkpoint(4, contact_clear=False))
    assert not buffer.consider(checkpoint(4, reliability=.2))
    assert buffer.consider(checkpoint(4))


def test_selector_uses_diagnosis_compatible_phase_and_can_step_back_deeper():
    checkpoints = [checkpoint(3, "observe"), checkpoint(9, "pregrasp"),
                   checkpoint(15, "transport")]
    selector = CheckpointSelector()
    first = selector.select(
        checkpoints, diagnosis="object_loss_risk", current_eef_pos=(.2, 0, .3),
        current_phase="transport", before_action_index=20)
    second = selector.select(
        checkpoints, diagnosis="object_loss_risk", current_eef_pos=(.2, 0, .3),
        current_phase="transport", before_action_index=20, rollback_level=1)
    assert first.phase in {"observe", "approach", "pregrasp"}
    assert second is not None and second != first
    assert all(x.phase != "transport" for x in (first, second))


def test_historical_corridor_is_reverse_ordered_and_includes_target():
    history = [
        {"action_index": i, "eef_pos": [i / 100, 0, .3], "joint_pos": [i / 100] * 7,
         "recovery_safe": i != 8}
        for i in range(4, 11)
    ]
    waypoints = HistoricalCorridorPlanner(stride=2).build(
        history, target_action_index=4, current_action_index=11)
    assert [x.action_index for x in waypoints] == [10, 7, 5, 4]


def test_cartesian_controller_limits_step_and_preserves_gripper_state():
    controller = CartesianRollbackController(maximum_translation_m=.005)
    action = controller.next_action(
        [0, 0, 0], [.02, 0, 0], [0, 0, 0, 0, 0, 0, -1])
    assert np.allclose(action[:3], [.1, 0, 0])
    assert action[6] == -1
    reversed_action = controller.next_action(
        [0, 0, 0], [-.02, 0, 0], [0, 0, 0, 0, 0, 0, -1])
    # The reference may reverse, but the commanded displacement cannot jump
    # directly from +5 mm to -5 mm in one controller update.
    assert np.allclose(reversed_action[:3], [.06, 0, 0])


def test_safety_gate_rejects_occupied_corridor_and_bad_response():
    gate = RecoverySafetyGate()
    pre = gate.precheck(
        current_eef=[0, 0, 0], predicted_eef=[.005, 0, 0],
        current_joint=[0] * 7, predicted_joint=[.01] * 7,
        joint_margin=.2, singularity_sigma=.2, corridor_deviation=.01,
        occupancy_clear=False)
    assert not pre.safe and "visual_corridor_occupied_or_unknown" in pre.reasons
    post = gate.postcheck(
        commanded_delta=[.005, 0, 0], actual_delta=[0, .005, 0],
        contact_probability=.1, contact_confidence=.9)
    assert post.state == "replan" and "response_direction_mismatch" in post.reasons


def test_failed_plan_gate_requires_similarity_reentry_and_no_progress_together():
    gate = FailedPlanNoveltyGate()
    failed = [[.5, 0, 0, 0, 0, 0, 0]] * 5
    same = [[.49, 0, 0, 0, 0, 0, 0]] * 5
    repeated = gate.decide(
        failed, same, reenters_failure_region=True, predicted_recovery_progress=0.0)
    assert repeated.reject
    useful = gate.decide(
        failed, same, reenters_failure_region=True, predicted_recovery_progress=.2)
    assert not useful.reject
    different_region = gate.decide(
        failed, same, reenters_failure_region=False, predicted_recovery_progress=0.0)
    assert not different_region.reject


def test_repeated_plan_rejection_resamples_then_rolls_back_deeper_then_stops():
    escalation = RecoveryEscalationController(
        RecoveryEscalationConfig(maximum_candidate_retries_per_level=2,
                                 maximum_rollback_levels=2))
    assert escalation.candidate_evaluated(accepted=False).action == "resample_candidate"
    deeper = escalation.candidate_evaluated(accepted=False)
    assert deeper.action == "rollback_deeper" and deeper.rollback_level == 1
    assert escalation.candidate_evaluated(accepted=False).action == "resample_candidate"
    assert escalation.candidate_evaluated(accepted=False).action == "safe_stop"


def test_windowed_response_monitor_tolerates_transient_then_checks_cumulative_motion():
    monitor = WindowedRecoveryResponseMonitor(window=3)
    command = [.005, 0, 0]
    assert monitor.observe(commanded_delta=command, actual_delta=[-.001, 0, 0],
                           contact_probability=0, contact_confidence=0).state == "warming_up"
    assert monitor.observe(commanded_delta=command, actual_delta=[.002, 0, 0],
                           contact_probability=0, contact_confidence=0).state == "warming_up"
    final = monitor.observe(commanded_delta=command, actual_delta=[.004, 0, 0],
                            contact_probability=0, contact_confidence=0)
    assert final.safe


def test_windowed_response_monitor_never_delays_hard_fault():
    monitor = WindowedRecoveryResponseMonitor(window=3)
    result = monitor.observe(commanded_delta=[.005, 0, 0], actual_delta=[0, 0, 0],
                             contact_probability=0, contact_confidence=0, hard_fault=True)
    assert result.state == "safe_stop"
