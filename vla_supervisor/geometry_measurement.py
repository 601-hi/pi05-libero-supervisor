"""Embodiment-calibrated gripper projection for shadow goal measurements.

The geometry chain is public-sensor reproducible: robot forward kinematics,
an embodiment-specific tool transform, and camera calibration.  A finite pixel
projection is diagnostic only until an independent calibration supplies a
validated error radius for the exact acquisition domain.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .camera_geometry import (
    CameraCalibration, camera_model_fingerprint, project_world_point, tool_point_world,
)
from .goal_measurement import ValidatedImagePoint


@dataclass(frozen=True)
class ToolPointCalibration:
    calibration_id: str
    embodiment_id: str
    local_offset_m: tuple[float, float, float]
    error_radius_px: float | None
    validation_id: str | None

    @property
    def validated(self) -> bool:
        return bool(
            self.calibration_id
            and self.embodiment_id
            and self.validation_id
            and self.error_radius_px is not None
            and math.isfinite(self.error_radius_px)
            and self.error_radius_px >= 0
        )


@dataclass(frozen=True)
class GeometryProjectionResult:
    point: ValidatedImagePoint | None
    diagnostic_xy_px: tuple[float, float] | None
    reason: str


def quaternion_to_rotation(quaternion, *, convention: str) -> np.ndarray:
    """Convert an explicitly labelled unit quaternion into a rotation matrix."""
    q = np.asarray(quaternion, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all():
        raise ValueError("quaternion must contain four finite values")
    if convention == "xyzw":
        x, y, z, w = q
    elif convention == "wxyz":
        w, x, y, z = q
    else:
        raise ValueError("quaternion convention must be explicit: xyzw or wxyz")
    norm = float(np.linalg.norm(q))
    if norm < 1e-12:
        raise ValueError("zero quaternion")
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.asarray([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ])


def project_gripper_reference(
    *,
    eef_position,
    eef_quaternion,
    quaternion_convention: str,
    camera: CameraCalibration,
    tool: ToolPointCalibration,
    action_index: int,
    measurement_domain: str,
) -> GeometryProjectionResult:
    """Project a tool point and promote it only when its error is calibrated."""
    rotation = quaternion_to_rotation(eef_quaternion, convention=quaternion_convention)
    world = tool_point_world(eef_position, rotation, tool.local_offset_m)
    projected = project_world_point(world, camera)
    if projected.xy is None or not projected.visible:
        return GeometryProjectionResult(None, projected.xy, projected.reason)
    xy = projected.xy
    if not tool.validated:
        return GeometryProjectionResult(None, xy, "tool_projection_uncalibrated")
    if not measurement_domain:
        return GeometryProjectionResult(None, xy, "missing_measurement_domain")
    expected_domain = camera_model_fingerprint(camera) + ":tool:" + tool.calibration_id
    if measurement_domain != expected_domain:
        return GeometryProjectionResult(None, xy, "measurement_domain_mismatch")
    normalized = (xy[0] / camera.image_width, xy[1] / camera.image_height)
    radius = float(tool.error_radius_px) / math.hypot(camera.image_width, camera.image_height)
    point = ValidatedImagePoint(
        xy=normalized,
        error_radius=radius,
        action_index=int(action_index),
        domain=measurement_domain,
        semantic_role="fingertip_midpoint",
        target_id="",
        validation_id=str(tool.validation_id),
        visible=True,
    )
    return GeometryProjectionResult(point, xy, "validated_geometry_projection")
