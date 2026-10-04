from vla_supervisor.controlled_object_selector_v2 import (
    CandidateControlEvidence,
    ControlledObjectSelectorV2,
)


def evidence(x, y, residual=0.01, robot_overlap=0.0, robot_motion=0.0):
    return CandidateControlEvidence((x, y), residual, robot_overlap, robot_motion)


def test_static_background_never_locks():
    selector = ControlledObjectSelectorV2(confirmation_frames=2)
    selector.update(gripper_command=1.0, gripper_moving=True, gripper_xy=(100, 100), projection_uncertainty_px=0, candidates={1: evidence(105, 105, 0.0)}, background_confidence=1.0)
    result = selector.update(gripper_command=1.0, gripper_moving=True, gripper_xy=(105, 100), projection_uncertainty_px=0, candidates={1: evidence(105, 105, 0.0)}, background_confidence=1.0)
    assert result.controlled_object_id is None


def test_post_close_nearby_independent_motion_locks():
    selector = ControlledObjectSelectorV2(confirmation_frames=2, minimum_score_margin=0.001)
    selector.update(gripper_command=1.0, gripper_moving=True, gripper_xy=(100, 100), projection_uncertainty_px=5, candidates={1: evidence(104, 104)}, background_confidence=0.8)
    selector.update(gripper_command=1.0, gripper_moving=True, gripper_xy=(105, 100), projection_uncertainty_px=5, candidates={1: evidence(109, 104)}, background_confidence=0.8)
    result = selector.update(gripper_command=1.0, gripper_moving=True, gripper_xy=(110, 100), projection_uncertainty_px=5, candidates={1: evidence(114, 104)}, background_confidence=0.8)
    assert result.controlled_object_id == 1
    assert result.state == "controlled"


def test_far_moving_distractor_is_rejected():
    selector = ControlledObjectSelectorV2(confirmation_frames=1)
    selector.update(gripper_command=1.0, gripper_moving=True, gripper_xy=(20, 20), projection_uncertainty_px=0, candidates={2: evidence(180, 180)}, background_confidence=1.0)
    result = selector.update(gripper_command=1.0, gripper_moving=True, gripper_xy=(25, 20), projection_uncertainty_px=0, candidates={2: evidence(190, 180)}, background_confidence=1.0)
    assert result.controlled_object_id is None


def test_robot_body_candidate_is_rejected_even_if_synchronized():
    selector = ControlledObjectSelectorV2(confirmation_frames=1)
    selector.update(gripper_command=1.0, gripper_moving=True, gripper_xy=(100, 100), projection_uncertainty_px=0, candidates={3: evidence(102, 102, robot_overlap=0.9)}, background_confidence=1.0)
    result = selector.update(gripper_command=1.0, gripper_moving=True, gripper_xy=(110, 100), projection_uncertainty_px=0, candidates={3: evidence(112, 102, robot_overlap=0.9)}, background_confidence=1.0)
    assert result.controlled_object_id is None


def test_candidate_that_moved_with_robot_before_close_is_rejected():
    selector = ControlledObjectSelectorV2(confirmation_frames=1)
    selector.update(gripper_command=1.0, gripper_moving=True, gripper_xy=(100, 100), projection_uncertainty_px=0, candidates={3: evidence(102, 102, robot_motion=0.95)}, background_confidence=1.0)
    result = selector.update(gripper_command=1.0, gripper_moving=True, gripper_xy=(110, 100), projection_uncertainty_px=0, candidates={3: evidence(112, 102, robot_motion=0.95)}, background_confidence=1.0)
    assert result.controlled_object_id is None


def test_invalid_background_causes_unknown_not_false_lock():
    selector = ControlledObjectSelectorV2()
    result = selector.update(gripper_command=1.0, gripper_moving=True, gripper_xy=(100, 100), projection_uncertainty_px=0, candidates={1: evidence(101, 101)}, background_confidence=0.0)
    assert result.state == "unknown"
    assert not result.evidence_valid


def test_open_command_resets_locked_identity():
    selector = ControlledObjectSelectorV2(confirmation_frames=1, minimum_score_margin=0.001)
    selector.update(gripper_command=1.0, gripper_moving=True, gripper_xy=(100, 100), projection_uncertainty_px=0, candidates={1: evidence(100, 100)}, background_confidence=1.0)
    selector.update(gripper_command=1.0, gripper_moving=True, gripper_xy=(110, 100), projection_uncertainty_px=0, candidates={1: evidence(110, 100)}, background_confidence=1.0)
    result = selector.update(gripper_command=-1.0, gripper_moving=False, gripper_xy=(110, 100), projection_uncertainty_px=0, candidates={}, background_confidence=1.0)
    assert result.controlled_object_id is None
    assert result.state == "open"
