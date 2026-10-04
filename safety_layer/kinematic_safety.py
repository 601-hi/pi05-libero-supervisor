"""Deployable kinematic safety indicators for a serial manipulator.

The core is simulator agnostic: callers provide an end-effector Jacobian,
joint positions, and joint limits.  A robosuite adapter may obtain those
quantities from the controller, but the same quantities are available from a
real robot model and joint encoders.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class KinematicSafetyMetrics:
    sigma_min: float
    sigma_max: float
    condition_number: float
    manipulability_log: float
    translation_sigma_min: float
    translation_sigma_max: float
    translation_condition_number: float
    rotation_sigma_min: float
    rotation_sigma_max: float
    rotation_condition_number: float
    joint_limit_margin_rad: float
    joint_limit_margin_fraction: float
    near_singular: bool
    near_joint_limit: bool
    invalid: bool

    def to_dict(self) -> dict:
        return asdict(self)


def compute_kinematic_safety(
    jacobian: Sequence[Sequence[float]],
    joint_position: Sequence[float],
    joint_limits: Sequence[Sequence[float]],
    *,
    sigma_min_threshold: float = 0.05,
    condition_threshold: float = 100.0,
    joint_margin_fraction_threshold: float = 0.05,
    eps: float = 1e-12,
) -> KinematicSafetyMetrics:
    """Compute singularity and joint-limit indicators without privileged state.

    ``jacobian`` may be 3xn (translation only) or 6xn (spatial twist).  Because
    translational and angular rows use different units, thresholds must be
    calibrated separately for the chosen representation and robot model.
    """

    j = np.asarray(jacobian, dtype=np.float64)
    q = np.asarray(joint_position, dtype=np.float64).reshape(-1)
    limits = np.asarray(joint_limits, dtype=np.float64)
    if j.ndim != 2 or j.shape[1] != q.size:
        raise ValueError("jacobian must have shape (m, number_of_joints)")
    if limits.shape != (q.size, 2):
        raise ValueError("joint_limits must have shape (number_of_joints, 2)")

    invalid = not (np.isfinite(j).all() and np.isfinite(q).all() and np.isfinite(limits).all())
    if invalid:
        return KinematicSafetyMetrics(*(math.nan,) * 12, True, True, True)

    def spectrum(matrix: np.ndarray) -> tuple[np.ndarray, float, float, float]:
        values = np.linalg.svd(matrix, compute_uv=False)
        maximum = float(values[0]) if values.size else 0.0
        minimum = float(values[-1]) if values.size else 0.0
        return values, minimum, maximum, float(maximum / max(minimum, eps))

    singular_values, sigma_min, sigma_max, condition = spectrum(j)
    if j.shape[0] == 6:
        _, translation_sigma_min, translation_sigma_max, translation_condition = spectrum(j[:3])
        _, rotation_sigma_min, rotation_sigma_max, rotation_condition = spectrum(j[3:])
    elif j.shape[0] == 3:
        translation_sigma_min = sigma_min
        translation_sigma_max = sigma_max
        translation_condition = condition
        rotation_sigma_min = math.nan
        rotation_sigma_max = math.nan
        rotation_condition = math.nan
    else:
        translation_sigma_min = math.nan
        translation_sigma_max = math.nan
        translation_condition = math.nan
        rotation_sigma_min = math.nan
        rotation_sigma_max = math.nan
        rotation_condition = math.nan

    # log product is numerically stable and equals log(sqrt(det(J J^T)))
    # for a full-row-rank Jacobian.
    manipulability_log = float(np.log(np.maximum(singular_values, eps)).sum())

    lower, upper = limits[:, 0], limits[:, 1]
    span = upper - lower
    bounded = span > eps
    if bounded.any():
        lower_margin = q[bounded] - lower[bounded]
        upper_margin = upper[bounded] - q[bounded]
        margins = np.minimum(lower_margin, upper_margin)
        margin_rad = float(margins.min())
        margin_fraction = float((margins / span[bounded]).min())
    else:
        margin_rad = math.inf
        margin_fraction = math.inf

    return KinematicSafetyMetrics(
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        condition_number=condition,
        manipulability_log=manipulability_log,
        translation_sigma_min=translation_sigma_min,
        translation_sigma_max=translation_sigma_max,
        translation_condition_number=translation_condition,
        rotation_sigma_min=rotation_sigma_min,
        rotation_sigma_max=rotation_sigma_max,
        rotation_condition_number=rotation_condition,
        joint_limit_margin_rad=margin_rad,
        joint_limit_margin_fraction=margin_fraction,
        near_singular=sigma_min < sigma_min_threshold or condition > condition_threshold,
        near_joint_limit=margin_fraction < joint_margin_fraction_threshold,
        invalid=False,
    )


def from_robosuite_robot(robot, **thresholds) -> KinematicSafetyMetrics:
    """Read deployable-equivalent quantities from a robosuite single arm."""

    controller = robot.controller
    controller.update()
    joint_indices = np.asarray(robot._ref_joint_indexes, dtype=int)
    joint_limits = np.asarray(robot.sim.model.jnt_range[joint_indices], dtype=np.float64)
    return compute_kinematic_safety(
        controller.J_full,
        robot._joint_positions,
        joint_limits,
        **thresholds,
    )
