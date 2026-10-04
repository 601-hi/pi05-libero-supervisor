"""MuJoCo-only collision oracle for validating deployable risk proxies.

The oracle reads simulator geometry unavailable on a real robot.  Its outputs
are evaluation labels and must never be presented as deployable sensor input.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import mujoco
import numpy as np


@dataclass(frozen=True)
class SweptDistanceResult:
    minimum_self_clearance_m: float | None
    minimum_environment_clearance_m: float | None
    self_pair: tuple[str, str] | None
    environment_pair: tuple[str, str] | None
    interpolation_samples: int
    baseline_self_clearance_m: float | None
    final_self_clearance_m: float | None
    baseline_environment_clearance_m: float | None
    final_environment_clearance_m: float | None
    baseline_self_pair_clearances: tuple[tuple[str, str, float], ...] = ()
    final_self_pair_clearances: tuple[tuple[str, str, float], ...] = ()
    baseline_environment_pair_clearances: tuple[tuple[str, str, float], ...] = ()
    final_environment_pair_clearances: tuple[tuple[str, str, float], ...] = ()
    # FK evidence sampled at exactly the same joint states as the collision
    # sweep.  This is oracle evidence for evaluation and candidate review; it
    # does not change the joint-space command sent to the controller.
    eef_positions_m: tuple[tuple[float, float, float], ...] = ()
    oracle_only: bool = True


def predict_joint_delta_from_translation(sim, robot, translation_delta: Sequence[float],
                                         damping: float = 0.05) -> np.ndarray:
    """Damped least-squares local IK prediction, without changing simulation."""
    controller = robot.controller
    velocity_indexes = np.asarray(controller.qvel_index, dtype=int)
    jacobian = np.asarray(sim.data.get_site_jacp(controller.eef_name), dtype=float)
    jacobian = jacobian.reshape(3, -1)[:, velocity_indexes]
    delta = np.asarray(translation_delta, dtype=float)
    if delta.shape != (3,):
        raise ValueError("translation_delta must have shape (3,)")
    regularized = jacobian @ jacobian.T + float(damping) ** 2 * np.eye(3)
    return jacobian.T @ np.linalg.solve(regularized, delta)


def damped_pose_joint_delta(jacobian: np.ndarray, pose_delta: Sequence[float], *,
                            damping: float = .05,
                            current_joint: Sequence[float] | None = None,
                            target_joint: Sequence[float] | None = None,
                            nullspace_gain: float = .15,
                            maximum_joint_delta_rad: float = .08) -> np.ndarray:
    """Solve a 6D local task and optionally bias redundant joints in nullspace."""
    jacobian = np.asarray(jacobian, dtype=float)
    delta = np.asarray(pose_delta, dtype=float)
    if jacobian.ndim != 2 or delta.shape != (jacobian.shape[0],):
        raise ValueError("pose_delta dimension must match Jacobian rows")
    regularized = jacobian @ jacobian.T + float(damping) ** 2 * np.eye(jacobian.shape[0])
    pseudoinverse = jacobian.T @ np.linalg.solve(regularized, np.eye(jacobian.shape[0]))
    joint_delta = pseudoinverse @ delta
    if (current_joint is None) != (target_joint is None):
        raise ValueError("current_joint and target_joint must be supplied together")
    if current_joint is not None:
        current = np.asarray(current_joint, dtype=float)
        target = np.asarray(target_joint, dtype=float)
        if current.shape != (jacobian.shape[1],) or target.shape != current.shape:
            raise ValueError("joint targets must match Jacobian columns")
        nullspace = np.eye(jacobian.shape[1]) - pseudoinverse @ jacobian
        joint_delta += float(nullspace_gain) * nullspace @ (target - current)
    maximum = float(np.max(np.abs(joint_delta))) if joint_delta.size else 0.0
    if maximum > maximum_joint_delta_rad:
        joint_delta *= maximum_joint_delta_rad / maximum
    return joint_delta


def predict_joint_delta_from_pose(sim, robot, translation_delta: Sequence[float],
                                  rotation_delta: Sequence[float], *,
                                  target_joint: Sequence[float] | None = None,
                                  damping: float = .05,
                                  nullspace_gain: float = .15,
                                  maximum_joint_delta_rad: float = .08) -> np.ndarray:
    """6D damped local IK plus optional history-seeking nullspace motion."""
    controller = robot.controller
    velocity_indexes = np.asarray(controller.qvel_index, dtype=int)
    jacobian_p = np.asarray(
        sim.data.get_site_jacp(controller.eef_name), dtype=float).reshape(3, -1)
    jacobian_r = np.asarray(
        sim.data.get_site_jacr(controller.eef_name), dtype=float).reshape(3, -1)
    jacobian = np.vstack((jacobian_p[:, velocity_indexes], jacobian_r[:, velocity_indexes]))
    translation = np.asarray(translation_delta, dtype=float)
    rotation = np.asarray(rotation_delta, dtype=float)
    if translation.shape != (3,) or rotation.shape != (3,):
        raise ValueError("translation_delta and rotation_delta must have shape (3,)")
    current = np.asarray(sim.data.qpos[
        np.asarray(robot._ref_joint_pos_indexes, dtype=int)], dtype=float)
    return damped_pose_joint_delta(
        jacobian, np.concatenate((translation, rotation)), damping=damping,
        current_joint=current if target_joint is not None else None,
        target_joint=target_joint, nullspace_gain=nullspace_gain,
        maximum_joint_delta_rad=maximum_joint_delta_rad)


class MuJoCoSweptCollisionOracle:
    """Query signed geom distances along a joint interpolation and restore state."""

    def __init__(self, sim, robot, *, interpolation_samples: int = 7,
                 distance_cap_m: float = 1.0,
                 pair_tracking_distance_m: float = .01):
        if interpolation_samples < 2:
            raise ValueError("interpolation_samples must be at least two")
        self.sim = sim
        self.robot = robot
        self.interpolation_samples = int(interpolation_samples)
        self.distance_cap_m = float(distance_cap_m)
        self.pair_tracking_distance_m = float(pair_tracking_distance_m)
        self.robot_geoms, self.environment_geoms = self._classify_geoms()

    def _eef_position(self) -> tuple[float, float, float]:
        site_name = self.robot.gripper.important_sites["grip_site"]
        position = np.asarray(self.sim.data.get_site_xpos(site_name), dtype=float)
        return tuple(float(value) for value in position)

    def _classify_geoms(self):
        model = self.sim.model
        robot_geoms = []
        environment_geoms = []
        # ``MjModel.geom_names`` in the robosuite / mujoco-py binding is a
        # compact name table, not an ID-indexed array.  Enumerating it silently
        # associates names with the wrong geom IDs.  Always resolve each real
        # MuJoCo ID through the binding API.
        for geom_id in range(int(model.ngeom)):
            name = model.geom_id2name(geom_id) or f"geom_{geom_id}"
            # MuJoCo / robosuite also carries non-contact proxy geoms with
            # conaffinity but contype=0 (for example full finger meshes).  They
            # can geometrically overlap the table while the smaller pad geom is
            # the actual contact shape, so only contact-producing geoms belong
            # in the distance oracle.
            collidable = bool(model.geom_contype[geom_id])
            if not collidable:
                continue
            if name == "robot0_link0_collision":
                # The fixed base collision mesh intentionally overlaps the
                # pedestal it is bolted to and cannot be changed by an arm
                # escape candidate.  Moving links remain included so an arm
                # sweep into the pedestal is still visible.
                continue
            if name.startswith("robot0_") or name.startswith("gripper0_"):
                robot_geoms.append(geom_id)
            else:
                environment_geoms.append(geom_id)
        return tuple(robot_geoms), tuple(environment_geoms)

    def _body_graph_distance(self, body_a: int, body_b: int) -> int | None:
        model = self.sim.model
        ancestors = {}
        cursor = int(body_a)
        distance = 0
        while cursor >= 0:
            ancestors[cursor] = distance
            if cursor == 0:
                break
            cursor = int(model.body_parentid[cursor])
            distance += 1
        cursor = int(body_b)
        distance = 0
        while cursor >= 0:
            if cursor in ancestors:
                return distance + ancestors[cursor]
            if cursor == 0:
                break
            cursor = int(model.body_parentid[cursor])
            distance += 1
        return None

    def _kinematically_near(self, geom_a: int, geom_b: int) -> bool:
        model = self.sim.model
        distance = self._body_graph_distance(
            int(model.geom_bodyid[geom_a]), int(model.geom_bodyid[geom_b]))
        # Neighboring link collision meshes intentionally overlap around their
        # joints.  Three body-tree edges also covers link7 -> hand and finger
        # attachment chains without suppressing nonlocal arm self-collision.
        return distance is not None and distance <= 3

    def _same_gripper_internal_pair(self, geom_a: int, geom_b: int) -> bool:
        """Whether both geoms belong to the same articulated gripper.

        Opposing finger pads are designed to approach during closure and
        grasping. Treating their small separation as arm self-collision makes
        every legitimate closed-gripper state fail the recovery gate.
        """
        model = self.sim.model
        name_a = model.geom_id2name(int(geom_a)) or ""
        name_b = model.geom_id2name(int(geom_b)) or ""
        return name_a.startswith("gripper0_") and name_b.startswith("gripper0_")

    def _body_has_free_joint(self, body_id: int) -> bool:
        """Whether a geom belongs to a body tree rooted in a free object.

        MuJoCo joint type 0 is ``mjJNT_FREE``.  Object meshes may live on a
        child body below the body carrying the free joint, so inspect every
        ancestor rather than only the geom's immediate body.
        """
        model = self.sim.model
        cursor = int(body_id)
        while cursor > 0:
            joint_start = int(model.body_jntadr[cursor])
            joint_count = int(model.body_jntnum[cursor])
            for joint_id in range(joint_start, joint_start + joint_count):
                if int(model.jnt_type[joint_id]) == 0:
                    return True
            cursor = int(model.body_parentid[cursor])
        return False

    def _allowed_manipulation_pair(self, robot_geom: int, environment_geom: int) -> bool:
        """Exclude expected finger-to-free-object approach from fixed risk.

        This is intentionally narrower than ignoring movable objects: arm
        links and the gripper palm can still collide with them, while fingers
        and pads are allowed to approach an object that is physically free to
        be manipulated.  Fixed furniture and fixtures are never exempted.
        """
        model = self.sim.model
        robot_name = model.geom_id2name(int(robot_geom)) or ""
        if not robot_name.startswith("gripper0_finger"):
            return False
        body_id = int(model.geom_bodyid[int(environment_geom)])
        return self._body_has_free_joint(body_id)

    def _distance(self, geom_a: int, geom_b: int) -> float:
        return float(mujoco.mj_geomDistance(
            self.sim.model._model, self.sim.data._data,
            int(geom_a), int(geom_b), self.distance_cap_m, None))

    def _minimum_distances(self):
        model = self.sim.model
        minimum_self = None
        minimum_self_pair = None
        for offset, geom_a in enumerate(self.robot_geoms):
            for geom_b in self.robot_geoms[offset + 1:]:
                if (self._kinematically_near(geom_a, geom_b)
                        or self._same_gripper_internal_pair(geom_a, geom_b)):
                    continue
                distance = self._distance(geom_a, geom_b)
                if minimum_self is None or distance < minimum_self:
                    minimum_self = distance
                    minimum_self_pair = (model.geom_id2name(geom_a), model.geom_id2name(geom_b))
        minimum_environment = None
        minimum_environment_pair = None
        for geom_a in self.robot_geoms:
            for geom_b in self.environment_geoms:
                if self._allowed_manipulation_pair(geom_a, geom_b):
                    continue
                distance = self._distance(geom_a, geom_b)
                if minimum_environment is None or distance < minimum_environment:
                    minimum_environment = distance
                    minimum_environment_pair = (
                        model.geom_id2name(geom_a), model.geom_id2name(geom_b))
        return minimum_self, minimum_environment, minimum_self_pair, minimum_environment_pair

    def _tracked_pair_clearances(self):
        """Return every pair close enough to matter, not only the minimum pair."""
        model = self.sim.model
        threshold = self.pair_tracking_distance_m
        self_rows = []
        environment_rows = []
        for offset, geom_a in enumerate(self.robot_geoms):
            for geom_b in self.robot_geoms[offset + 1:]:
                if (self._kinematically_near(geom_a, geom_b)
                        or self._same_gripper_internal_pair(geom_a, geom_b)):
                    continue
                distance = self._distance(geom_a, geom_b)
                if distance <= threshold:
                    self_rows.append((model.geom_id2name(geom_a),
                                      model.geom_id2name(geom_b), distance))
            for geom_b in self.environment_geoms:
                if self._allowed_manipulation_pair(geom_a, geom_b):
                    continue
                distance = self._distance(geom_a, geom_b)
                if distance <= threshold:
                    environment_rows.append((model.geom_id2name(geom_a),
                                             model.geom_id2name(geom_b), distance))
        return tuple(sorted(self_rows)), tuple(sorted(environment_rows))

    def current_clearances(self, *, include_tracked_pairs: bool = False) -> SweptDistanceResult:
        """Read current signed clearances without forwarding or restoring MuJoCo.

        Checkpoint admission calls this on every nominal step.  Using
        ``evaluate(current_joint)`` would perform two ``sim.forward`` calls and
        can introduce floating-point state drift that is later amplified by a
        closed-loop policy, even though no intervention occurred.
        """
        self_distance, environment_distance, self_pair, environment_pair = (
            self._minimum_distances())
        tracked_self, tracked_environment = (
            self._tracked_pair_clearances()
            if include_tracked_pairs and hasattr(self, "robot_geoms") else ((), ()))
        return SweptDistanceResult(
            minimum_self_clearance_m=self_distance,
            minimum_environment_clearance_m=environment_distance,
            self_pair=self_pair,
            environment_pair=environment_pair,
            interpolation_samples=1,
            baseline_self_clearance_m=self_distance,
            final_self_clearance_m=self_distance,
            baseline_environment_clearance_m=environment_distance,
            final_environment_clearance_m=environment_distance,
            baseline_self_pair_clearances=tracked_self,
            final_self_pair_clearances=tracked_self,
            baseline_environment_pair_clearances=tracked_environment,
            final_environment_pair_clearances=tracked_environment,
            eef_positions_m=(self._eef_position(),),
        )

    def evaluate(self, predicted_joint: Sequence[float]) -> SweptDistanceResult:
        predicted = np.asarray(predicted_joint, dtype=float)
        qpos_indexes = np.asarray(self.robot._ref_joint_pos_indexes, dtype=int)
        current = np.asarray(self.sim.data.qpos[qpos_indexes], dtype=float).copy()
        if predicted.shape != current.shape:
            raise ValueError("predicted_joint shape does not match robot arm joints")
        saved = self.sim.get_state()
        min_self = None
        min_environment = None
        self_pair = None
        environment_pair = None
        baseline_self = None
        final_self = None
        baseline_environment = None
        final_environment = None
        baseline_self_pairs = ()
        final_self_pairs = ()
        baseline_environment_pairs = ()
        final_environment_pairs = ()
        eef_positions = []
        try:
            fractions = np.linspace(0.0, 1.0, self.interpolation_samples)
            for sample_index, fraction in enumerate(fractions):
                self.sim.data.qpos[qpos_indexes] = current + fraction * (predicted - current)
                self.sim.forward()
                eef_positions.append(self._eef_position())
                values = self._minimum_distances()
                if sample_index == 0:
                    baseline_self, baseline_environment = values[0], values[1]
                    baseline_self_pairs, baseline_environment_pairs = (
                        self._tracked_pair_clearances())
                if sample_index == len(fractions) - 1:
                    final_self, final_environment = values[0], values[1]
                    final_self_pairs, final_environment_pairs = (
                        self._tracked_pair_clearances())
                if min_self is None or values[0] < min_self:
                    min_self, self_pair = values[0], values[2]
                if min_environment is None or values[1] < min_environment:
                    min_environment, environment_pair = values[1], values[3]
        finally:
            self.sim.set_state(saved)
            self.sim.forward()
        return SweptDistanceResult(
            minimum_self_clearance_m=min_self,
            minimum_environment_clearance_m=min_environment,
            self_pair=self_pair,
            environment_pair=environment_pair,
            interpolation_samples=self.interpolation_samples,
            baseline_self_clearance_m=baseline_self,
            final_self_clearance_m=final_self,
            baseline_environment_clearance_m=baseline_environment,
            final_environment_clearance_m=final_environment,
            baseline_self_pair_clearances=baseline_self_pairs,
            final_self_pair_clearances=final_self_pairs,
            baseline_environment_pair_clearances=baseline_environment_pairs,
            final_environment_pair_clearances=final_environment_pairs,
            eef_positions_m=tuple(eef_positions),
        )
