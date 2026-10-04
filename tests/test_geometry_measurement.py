import math

import numpy as np
import pytest

from vla_supervisor.camera_geometry import CameraCalibration, camera_model_fingerprint
from vla_supervisor.geometry_measurement import (
    ToolPointCalibration, project_gripper_reference, quaternion_to_rotation,
)


def camera():
    return CameraCalibration("cam-cal", "fixed", ((100,0,50),(0,100,50),(0,0,1)),
        ((1,0,0,0),(0,1,0,0),(0,0,1,0),(0,0,0,1)), 100, 100)


def tool(**changes):
    values = dict(calibration_id="tool-cal", embodiment_id="panda",
        local_offset_m=(0,0,0), error_radius_px=5, validation_id="independent-cal-v1")
    values.update(changes)
    return ToolPointCalibration(**values)


def domain(cam=None, tool_cal=None):
    cam = cam or camera()
    tool_cal = tool_cal or tool()
    return camera_model_fingerprint(cam) + ":tool:" + tool_cal.calibration_id


def test_quaternion_convention_is_explicit_and_equivalent():
    assert np.allclose(quaternion_to_rotation((0,0,0,1), convention="xyzw"), np.eye(3))
    assert np.allclose(quaternion_to_rotation((1,0,0,0), convention="wxyz"), np.eye(3))
    with pytest.raises(ValueError):
        quaternion_to_rotation((0,0,0,1), convention="guess")


def test_validated_projection_becomes_normalized_measurement():
    result = project_gripper_reference(eef_position=(0,0,1), eef_quaternion=(0,0,0,1),
        quaternion_convention="xyzw", camera=camera(), tool=tool(), action_index=7,
        measurement_domain=domain())
    assert result.reason == "validated_geometry_projection"
    assert result.diagnostic_xy_px == pytest.approx((50,50))
    assert result.point.xy == pytest.approx((.5,.5))
    assert result.point.action_index == 7
    assert result.point.error_radius == pytest.approx(5 / math.hypot(100,100))


@pytest.mark.parametrize("change", [
    dict(validation_id=None), dict(error_radius_px=None), dict(error_radius_px=float("nan")),
    dict(error_radius_px=-1), dict(calibration_id=""), dict(embodiment_id=""),
])
def test_uncalibrated_geometry_remains_diagnostic_only(change):
    result = project_gripper_reference(eef_position=(0,0,1), eef_quaternion=(0,0,0,1),
        quaternion_convention="xyzw", camera=camera(), tool=tool(**change), action_index=0,
        measurement_domain=domain(tool_cal=tool(**change)))
    assert result.point is None
    assert result.diagnostic_xy_px == pytest.approx((50,50))
    assert result.reason == "tool_projection_uncalibrated"


def test_pose_and_scene_calibration_change_the_projection():
    offset = tool(local_offset_m=(.1,0,0))
    identity = project_gripper_reference(eef_position=(0,0,1), eef_quaternion=(0,0,0,1),
        quaternion_convention="xyzw", camera=camera(), tool=offset, action_index=0,
        measurement_domain=domain(tool_cal=offset)).diagnostic_xy_px
    quarter_turn = project_gripper_reference(eef_position=(0,0,1),
        eef_quaternion=(0,0,math.sin(math.pi/4),math.cos(math.pi/4)),
        quaternion_convention="xyzw", camera=camera(), tool=offset, action_index=0,
        measurement_domain=domain(tool_cal=offset)).diagnostic_xy_px
    assert identity == pytest.approx((60,50))
    assert quarter_turn == pytest.approx((50,60))


def test_points_behind_camera_never_become_measurements():
    result = project_gripper_reference(eef_position=(0,0,-1), eef_quaternion=(0,0,0,1),
        quaternion_convention="xyzw", camera=camera(), tool=tool(), action_index=0,
        measurement_domain=domain())
    assert result.point is None and result.reason == "behind_camera"


def test_caller_cannot_relabel_a_projection_as_another_domain():
    result = project_gripper_reference(eef_position=(0,0,1), eef_quaternion=(0,0,0,1),
        quaternion_convention="xyzw", camera=camera(), tool=tool(), action_index=0,
        measurement_domain="claimed-other-domain")
    assert result.point is None and result.reason == "measurement_domain_mismatch"
