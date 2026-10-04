from vla_supervisor.grasp_control_state import GraspControlMonitor, GraspControlState


def update(monitor, *, command=1.0, moving=True, near=(4,), independent=(4,), sync=(4,), valid=True):
    return monitor.update(
        gripper_command=command, gripper_moving=moving,
        nearby_candidate_ids=near, independently_moving_ids=independent,
        synchronized_with_gripper_ids=sync, evidence_valid=valid,
    )


def test_sustained_unique_coupling_establishes_control():
    monitor = GraspControlMonitor(control_confirmation_frames=3)
    assert update(monitor).state is GraspControlState.CONTROL_CANDIDATE
    assert update(monitor).state is GraspControlState.CONTROL_CANDIDATE
    result = update(monitor)
    assert result.state is GraspControlState.CONTROLLED
    assert result.controlled_object_id == 4


def test_grace_counts_only_informative_gripper_motion():
    monitor = GraspControlMonitor(effect_grace_motion_frames=3)
    for _ in range(10):
        assert update(monitor, moving=False, independent=(), sync=()).state is GraspControlState.WAITING_FOR_EFFECT
    for _ in range(2):
        assert update(monitor, moving=True, independent=(), sync=()).state is GraspControlState.WAITING_FOR_EFFECT
    assert update(monitor, moving=True, independent=(), sync=()).state is GraspControlState.GRASP_NOT_ESTABLISHED


def test_motion_without_synchrony_is_not_control():
    monitor = GraspControlMonitor(effect_grace_motion_frames=3)
    assert update(monitor, sync=()).state is GraspControlState.MOVED_NOT_CONTROLLED
    assert update(monitor, sync=()).state is GraspControlState.MOVED_NOT_CONTROLLED
    assert update(monitor, sync=()).state is GraspControlState.GRASP_NOT_ESTABLISHED


def test_invalid_evidence_does_not_consume_grace():
    monitor = GraspControlMonitor(effect_grace_motion_frames=2)
    assert update(monitor, valid=False).state is GraspControlState.UNKNOWN
    assert update(monitor, valid=False).informative_motion_frames == 0
    assert update(monitor, independent=(), sync=()).state is GraspControlState.WAITING_FOR_EFFECT


def test_control_loss_requires_motion_evidence_and_open_rearms():
    monitor = GraspControlMonitor(control_confirmation_frames=1, control_loss_motion_frames=2)
    assert update(monitor).state is GraspControlState.CONTROLLED
    assert update(monitor, moving=False, independent=(), sync=()).state is GraspControlState.CONTROLLED
    assert update(monitor, independent=(), sync=()).state is GraspControlState.CONTROLLED
    assert update(monitor, independent=(), sync=()).state is GraspControlState.CONTROL_LOST
    assert update(monitor, command=-1.0, independent=(), sync=()).state is GraspControlState.OPEN
