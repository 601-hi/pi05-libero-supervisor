"""Public robot-model measurements used by checkpoint and recovery safety."""
from __future__ import annotations

import numpy as np


def measure_public_kinematic_safety(sim, robot) -> dict:
    """Return joint-limit margin and translational Jacobian conditioning.

    This uses only the robot/controller model available on a real deployment
    (joint limits, encoders and the calibrated kinematic chain).  Failures are
    explicit and must not be interpreted as a safe reading.
    """
    result = {
        "joint_margin": None,
        "singularity_sigma": None,
        "kinematic_measurement_status": "unavailable",
    }
    try:
        qpos_indexes = np.asarray(robot._ref_joint_pos_indexes, dtype=int)
        joint_indexes = np.asarray(robot._ref_joint_indexes, dtype=int)
        qpos = np.asarray(sim.data.qpos[qpos_indexes], dtype=float)
        ranges = np.asarray(sim.model.jnt_range[joint_indexes], dtype=float)
        limited = np.asarray(sim.model.jnt_limited[joint_indexes], dtype=bool)
        if np.any(limited):
            lower_margin = qpos[limited] - ranges[limited, 0]
            upper_margin = ranges[limited, 1] - qpos[limited]
            result["joint_margin"] = float(np.min(np.minimum(lower_margin, upper_margin)))

        controller = robot.controller
        jacobian = np.asarray(
            sim.data.get_site_jacp(controller.eef_name), dtype=float
        ).reshape(3, -1)[:, np.asarray(controller.qvel_index, dtype=int)]
        singular_values = np.linalg.svd(jacobian, compute_uv=False)
        result["singularity_sigma"] = float(np.min(singular_values))
        result["kinematic_measurement_status"] = "ok"
    except (AttributeError, IndexError, TypeError, ValueError, np.linalg.LinAlgError) as exc:
        result["kinematic_measurement_status"] = f"unavailable:{type(exc).__name__}"
    return result

