"""Task-agnostic checkpoint rollback and failed-plan rejection primitives.

This module plans reference recovery motion; it does not replace the robot
controller or physics engine.  All outputs remain subject to the normal
instruction guard and online pre/post execution safety checks.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from typing import Iterable, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class RecoveryCheckpoint:
    action_index: int
    eef_pos: tuple[float, float, float]
    joint_pos: tuple[float, ...]
    phase: str
    observability: float
    reliability: float
    contact_clear: bool
    joint_margin: float
    singularity_margin: float
    task_progress: float = 0.0


@dataclass(frozen=True)
class CheckpointAdmissionConfig:
    minimum_observability: float = 0.6
    minimum_reliability: float = 0.7
    minimum_joint_margin: float = 0.05
    minimum_singularity_margin: float = 0.03
    minimum_spacing_steps: int = 3


class CheckpointBuffer:
    """Store only states that passed deployable trust gates."""

    def __init__(self, config: CheckpointAdmissionConfig = CheckpointAdmissionConfig(),
                 max_checkpoints: int = 100):
        self.config = config
        self.max_checkpoints = int(max_checkpoints)
        self.checkpoints: list[RecoveryCheckpoint] = []

    def reset(self):
        self.checkpoints.clear()

    def consider(self, checkpoint: RecoveryCheckpoint) -> bool:
        cfg = self.config
        admissible = (
            checkpoint.contact_clear
            and checkpoint.observability >= cfg.minimum_observability
            and checkpoint.reliability >= cfg.minimum_reliability
            and checkpoint.joint_margin >= cfg.minimum_joint_margin
            and checkpoint.singularity_margin >= cfg.minimum_singularity_margin
        )
        if not admissible:
            return False
        if (self.checkpoints and checkpoint.action_index - self.checkpoints[-1].action_index
                < cfg.minimum_spacing_steps):
            return False
        self.checkpoints.append(checkpoint)
        if len(self.checkpoints) > self.max_checkpoints:
            del self.checkpoints[0]
        return True


@dataclass(frozen=True)
class CheckpointSelectionConfig:
    distance_weight: float = 2.0
    phase_rollback_weight: float = 2.0
    observability_weight: float = 3.0
    reliability_weight: float = 2.0
    progress_preservation_weight: float = 1.0


_PHASE_ORDER = {
    "observe": 0,
    "approach": 1,
    "pregrasp": 2,
    "grasp": 3,
    "transport": 4,
    "place": 5,
    "goal_relation": 6,
}

_DIAGNOSIS_PHASES = {
    "fixed_obstacle_or_jam": {"observe", "approach", "pregrasp", "transport"},
    "empty_grasp_or_miss": {"observe", "approach", "pregrasp"},
    "wrong_object_control": {"observe", "approach", "pregrasp"},
    "object_loss_risk": {"observe", "approach", "pregrasp"},
    "approach_progress_stalled": {"observe", "approach"},
    "grasp_progress_stalled": {"approach", "pregrasp"},
    # A goal-relation failure is observed late, but a conservative recorder
    # may intentionally trust only the pre-contact prefix.  Recovery must be
    # allowed to fall back to those earlier safe states; otherwise the visual
    # diagnosis can never reach the rollback controller.
    "goal_relation_progress_stalled": {
        "observe", "approach", "pregrasp", "grasp", "transport", "place",
        "goal_relation",
    },
    "goal_relation_not_satisfied": {
        "observe", "approach", "pregrasp", "grasp", "transport", "place",
        "goal_relation",
    },
}


class CheckpointSelector:
    def __init__(self, config: CheckpointSelectionConfig = CheckpointSelectionConfig()):
        self.config = config

    def ranked(self, checkpoints: Iterable[RecoveryCheckpoint], *, diagnosis: str,
               current_eef_pos: Sequence[float], current_phase: str,
               before_action_index: int) -> list[RecoveryCheckpoint]:
        allowed = _DIAGNOSIS_PHASES.get(diagnosis, {"observe", "approach", "pregrasp"})
        current = np.asarray(current_eef_pos, dtype=float)
        current_rank = _PHASE_ORDER.get(current_phase, 0)

        def score(cp: RecoveryCheckpoint) -> float:
            distance = float(np.linalg.norm(np.asarray(cp.eef_pos) - current))
            rollback = max(0, current_rank - _PHASE_ORDER.get(cp.phase, 0))
            return (
                self.config.distance_weight * distance
                + self.config.phase_rollback_weight * rollback
                - self.config.observability_weight * cp.observability
                - self.config.reliability_weight * cp.reliability
                - self.config.progress_preservation_weight * cp.task_progress
            )

        candidates = [
            cp for cp in checkpoints
            if cp.action_index < before_action_index and cp.phase in allowed and cp.contact_clear
        ]
        return sorted(candidates, key=lambda cp: (score(cp), -cp.action_index))

    def select(self, checkpoints: Iterable[RecoveryCheckpoint], *, diagnosis: str,
               current_eef_pos: Sequence[float], current_phase: str,
               before_action_index: int, rollback_level: int = 0) -> RecoveryCheckpoint | None:
        ranked = self.ranked(
            checkpoints, diagnosis=diagnosis, current_eef_pos=current_eef_pos,
            current_phase=current_phase, before_action_index=before_action_index)
        return ranked[rollback_level] if rollback_level < len(ranked) else None


@dataclass(frozen=True)
class RecoveryWaypoint:
    action_index: int
    eef_pos: tuple[float, float, float]
    joint_pos: tuple[float, ...]


class HistoricalCorridorPlanner:
    """Turn previously observed safe states into reverse-ordered waypoints."""

    def __init__(self, stride: int = 3):
        if stride < 1:
            raise ValueError("stride must be positive")
        self.stride = int(stride)

    def build(self, history: Iterable[Mapping], *, target_action_index: int,
              current_action_index: int) -> tuple[RecoveryWaypoint, ...]:
        eligible = sorted((row for row in history
                           if target_action_index <= int(row["action_index"]) < current_action_index
                           and bool(row.get("recovery_safe", False))),
                          key=lambda row: int(row["action_index"]), reverse=True)
        selected = eligible[::self.stride]
        target = next((row for row in reversed(eligible)
                       if int(row["action_index"]) == target_action_index), None)
        if target is not None and (not selected or selected[-1] is not target):
            selected.append(target)
        return tuple(RecoveryWaypoint(
            int(row["action_index"]), tuple(float(x) for x in row["eef_pos"]),
            tuple(float(x) for x in row["joint_pos"])) for row in selected)


@dataclass(frozen=True)
class RecoverySafetyConfig:
    maximum_cartesian_step_m: float = 0.01
    maximum_rotation_step_rad: float = 0.10
    maximum_joint_step_rad: float = 0.12
    minimum_joint_margin_rad: float = 0.05
    minimum_singularity_sigma: float = 0.03
    maximum_corridor_deviation_rad: float = 0.15
    minimum_response_cosine: float = 0.3
    minimum_response_ratio: float = 0.1
    hard_contact_probability: float = 0.9


@dataclass(frozen=True)
class RecoverySafetyDecision:
    state: str
    reasons: tuple[str, ...]

    @property
    def safe(self) -> bool:
        return self.state == "safe"


class RecoverySafetyGate:
    def __init__(self, config: RecoverySafetyConfig = RecoverySafetyConfig()):
        self.config = config

    def precheck(self, *, current_eef, predicted_eef, current_joint, predicted_joint,
                 joint_margin: float, singularity_sigma: float,
                 corridor_deviation: float, occupancy_clear: bool,
                 rotation_step_rad: float = 0.0) -> RecoverySafetyDecision:
        cfg = self.config
        reasons = []
        if np.linalg.norm(np.asarray(predicted_eef) - np.asarray(current_eef)) > cfg.maximum_cartesian_step_m:
            reasons.append("cartesian_step_limit")
        if abs(float(rotation_step_rad)) > cfg.maximum_rotation_step_rad:
            reasons.append("rotation_step_limit")
        if np.max(np.abs(np.asarray(predicted_joint) - np.asarray(current_joint))) > cfg.maximum_joint_step_rad:
            reasons.append("joint_step_limit")
        if joint_margin < cfg.minimum_joint_margin_rad:
            reasons.append("joint_limit_margin")
        if singularity_sigma < cfg.minimum_singularity_sigma:
            reasons.append("singularity_margin")
        if corridor_deviation > cfg.maximum_corridor_deviation_rad:
            reasons.append("historical_corridor_deviation")
        if not occupancy_clear:
            reasons.append("visual_corridor_occupied_or_unknown")
        return RecoverySafetyDecision("safe" if not reasons else "replan", tuple(reasons))

    def postcheck(self, *, commanded_delta, actual_delta, contact_probability: float,
                  contact_confidence: float, hard_fault: bool = False) -> RecoverySafetyDecision:
        if hard_fault:
            return RecoverySafetyDecision("safe_stop", ("hard_robot_fault",))
        if (contact_confidence >= 0.5
                and contact_probability >= self.config.hard_contact_probability):
            return RecoverySafetyDecision("safe_stop", ("high_confidence_contact",))
        command = np.asarray(commanded_delta, dtype=float)
        actual = np.asarray(actual_delta, dtype=float)
        command_norm = float(np.linalg.norm(command))
        actual_norm = float(np.linalg.norm(actual))
        if command_norm <= 1e-9:
            return RecoverySafetyDecision("safe", ())
        cosine = float(command @ actual / (command_norm * actual_norm + 1e-12))
        ratio = actual_norm / command_norm
        reasons = []
        if cosine < self.config.minimum_response_cosine:
            reasons.append("response_direction_mismatch")
        if ratio < self.config.minimum_response_ratio:
            reasons.append("insufficient_response")
        return RecoverySafetyDecision("safe" if not reasons else "replan", tuple(reasons))


def split_pose_action_for_safety(action, *, translation_scale_m: float = 0.05,
                                 rotation_scale_rad: float = 0.5,
                                 maximum_translation_m: float = 0.01,
                                 maximum_rotation_rad: float = 0.10
                                 ) -> tuple[np.ndarray, ...]:
    """Split one normalized OSC pose action into bounded equivalent microsteps.

    Pose components preserve their clipped cumulative target; the gripper
    command is held constant.  This is a control bridge, not a new planner.
    Every returned microstep still requires fresh safety evidence.
    """
    value = np.asarray(action, dtype=float)
    if value.ndim != 1 or value.size < 7 or not np.all(np.isfinite(value)):
        raise ValueError("pose action must contain at least seven finite values")
    if min(translation_scale_m, rotation_scale_rad,
           maximum_translation_m, maximum_rotation_rad) <= 0:
        raise ValueError("action scales and microstep limits must be positive")
    clipped = value.copy()
    clipped[:6] = np.clip(clipped[:6], -1.0, 1.0)
    translation = translation_scale_m * clipped[:3]
    rotation = rotation_scale_rad * clipped[3:6]
    count = max(
        1,
        int(np.ceil(np.linalg.norm(translation) / maximum_translation_m)),
        int(np.ceil(np.linalg.norm(rotation) / maximum_rotation_rad)),
    )
    microstep = clipped.copy()
    microstep[:6] /= count
    return tuple(microstep.copy() for _ in range(count))


class WindowedRecoveryResponseMonitor:
    """Separate immediate hard faults from delayed closed-loop tracking checks."""

    def __init__(self, gate: RecoverySafetyGate | None = None, window: int = 3):
        if window < 1:
            raise ValueError("window must be positive")
        self.gate = gate or RecoverySafetyGate()
        self.window = int(window)
        self.reset()

    def reset(self):
        self.commands = deque(maxlen=self.window)
        self.actuals = deque(maxlen=self.window)
        self.rotation_commands = deque(maxlen=self.window)
        self.rotation_actuals = deque(maxlen=self.window)

    def observe(self, *, commanded_delta, actual_delta, contact_probability: float,
                contact_confidence: float, hard_fault: bool = False,
                commanded_rotation_delta=None,
                actual_rotation_delta=None) -> RecoverySafetyDecision:
        immediate = self.gate.postcheck(
            commanded_delta=[0, 0, 0], actual_delta=[0, 0, 0],
            contact_probability=contact_probability,
            contact_confidence=contact_confidence, hard_fault=hard_fault)
        if immediate.state == "safe_stop":
            return immediate
        self.commands.append(np.asarray(commanded_delta, dtype=float))
        self.actuals.append(np.asarray(actual_delta, dtype=float))
        if commanded_rotation_delta is not None or actual_rotation_delta is not None:
            if commanded_rotation_delta is None or actual_rotation_delta is None:
                raise ValueError("both commanded and actual rotation deltas are required")
            self.rotation_commands.append(np.asarray(commanded_rotation_delta, dtype=float))
            self.rotation_actuals.append(np.asarray(actual_rotation_delta, dtype=float))
        if len(self.commands) < self.window:
            return RecoverySafetyDecision("warming_up", ())
        translation = self.gate.postcheck(
            commanded_delta=np.sum(self.commands, axis=0),
            actual_delta=np.sum(self.actuals, axis=0),
            contact_probability=contact_probability,
            contact_confidence=contact_confidence, hard_fault=False)
        if not self.rotation_commands:
            return translation
        rotation = self.gate.postcheck(
            commanded_delta=np.sum(self.rotation_commands, axis=0),
            actual_delta=np.sum(self.rotation_actuals, axis=0),
            contact_probability=contact_probability,
            contact_confidence=contact_confidence, hard_fault=False)
        reasons = tuple(translation.reasons) + tuple(
            f"rotation_{reason}" for reason in rotation.reasons)
        return RecoverySafetyDecision("safe" if not reasons else "replan", reasons)

class CartesianRollbackController:
    """Generate one bounded LIBERO-style translation command toward a waypoint."""

    def __init__(self, maximum_translation_m: float = 0.005,
                 maximum_translation_change_m: float = 0.002,
                 action_scale_m: float = 0.05):
        self.maximum_translation_m = float(maximum_translation_m)
        self.maximum_translation_change_m = float(maximum_translation_change_m)
        self.action_scale_m = float(action_scale_m)
        self.reset()

    def reset(self):
        self.previous_delta = None

    def next_action(self, current_eef: Sequence[float], target_eef: Sequence[float],
                    reference_action: Sequence[float] | None = None) -> np.ndarray:
        delta = np.asarray(target_eef, float) - np.asarray(current_eef, float)
        norm = float(np.linalg.norm(delta))
        if norm > self.maximum_translation_m:
            delta *= self.maximum_translation_m / norm
        if self.previous_delta is not None:
            change = delta - self.previous_delta
            change_norm = float(np.linalg.norm(change))
            if change_norm > self.maximum_translation_change_m:
                delta = self.previous_delta + change * (
                    self.maximum_translation_change_m / change_norm)
        self.previous_delta = delta.copy()
        action = np.zeros(7, dtype=float)
        if reference_action is not None:
            reference = np.asarray(reference_action, dtype=float)
            action[6] = reference[6] if len(reference) > 6 else 0.0
        action[:3] = np.clip(delta / self.action_scale_m, -1.0, 1.0)
        return action


def quaternion_error_axis_angle(current_xyzw: Sequence[float],
                                target_xyzw: Sequence[float]) -> np.ndarray:
    """Shortest target * inverse(current) rotation as an axis-angle vector."""
    current = np.asarray(current_xyzw, dtype=float)
    target = np.asarray(target_xyzw, dtype=float)
    if current.shape != (4,) or target.shape != (4,):
        raise ValueError("quaternions must have shape (4,) in xyzw order")
    current_norm = float(np.linalg.norm(current))
    target_norm = float(np.linalg.norm(target))
    if current_norm <= 1e-12 or target_norm <= 1e-12:
        raise ValueError("quaternions must be nonzero")
    current = current / current_norm
    target = target / target_norm
    if float(current @ target) < 0:
        target = -target
    cx, cy, cz, cw = current
    tx, ty, tz, tw = target
    # Hamilton product target * conjugate(current), xyzw convention.
    vector = np.array([
        tw * -cx + tx * cw + ty * -cz - tz * -cy,
        tw * -cy - tx * -cz + ty * cw + tz * -cx,
        tw * -cz + tx * -cy - ty * -cx + tz * cw,
    ])
    scalar = tw * cw - tx * -cx - ty * -cy - tz * -cz
    vector_norm = float(np.linalg.norm(vector))
    if vector_norm <= 1e-12:
        return np.zeros(3)
    angle = 2.0 * np.arctan2(vector_norm, float(np.clip(scalar, -1.0, 1.0)))
    if angle > np.pi:
        angle -= 2.0 * np.pi
    return vector / vector_norm * angle


def apply_axis_angle_to_quaternion(current_xyzw: Sequence[float],
                                   axis_angle: Sequence[float]) -> np.ndarray:
    """Left-apply an axis-angle increment to an xyzw quaternion."""
    current = np.asarray(current_xyzw, dtype=float)
    rotation = np.asarray(axis_angle, dtype=float)
    if current.shape != (4,) or rotation.shape != (3,):
        raise ValueError("expected quaternion (4,) and axis-angle (3,)")
    current = current / max(float(np.linalg.norm(current)), 1e-12)
    angle = float(np.linalg.norm(rotation))
    if angle <= 1e-12:
        return current.copy()
    vector = rotation / angle * np.sin(angle / 2.0)
    scalar = np.cos(angle / 2.0)
    rx, ry, rz = vector
    rw = scalar
    cx, cy, cz, cw = current
    result = np.array([
        rw * cx + rx * cw + ry * cz - rz * cy,
        rw * cy - rx * cz + ry * cw + rz * cx,
        rw * cz + rx * cy - ry * cx + rz * cw,
        rw * cw - rx * cx - ry * cy - rz * cz,
    ])
    return result / max(float(np.linalg.norm(result)), 1e-12)


class PoseRollbackController:
    """Generate bounded LIBERO OSC_POSE commands toward a historical 6D pose."""

    def __init__(self, maximum_translation_m: float = .005,
                 maximum_rotation_rad: float = .05,
                 maximum_translation_change_m: float = .002,
                 maximum_rotation_change_rad: float = .025,
                 translation_action_scale_m: float = .05,
                 rotation_action_scale_rad: float = .5,
                 orientation_tolerance_rad: float = .01,
                 translation_integral_gain: float = 0.0,
                 translation_derivative_gain: float = 0.0,
                 rotation_integral_gain: float = 0.0,
                 rotation_derivative_gain: float = 0.0,
                 integral_limit: float = .01):
        self.maximum_translation_m = float(maximum_translation_m)
        self.maximum_rotation_rad = float(maximum_rotation_rad)
        self.maximum_translation_change_m = float(maximum_translation_change_m)
        self.maximum_rotation_change_rad = float(maximum_rotation_change_rad)
        self.translation_action_scale_m = float(translation_action_scale_m)
        self.rotation_action_scale_rad = float(rotation_action_scale_rad)
        self.orientation_tolerance_rad = float(orientation_tolerance_rad)
        self.translation_integral_gain = float(translation_integral_gain)
        self.translation_derivative_gain = float(translation_derivative_gain)
        self.rotation_integral_gain = float(rotation_integral_gain)
        self.rotation_derivative_gain = float(rotation_derivative_gain)
        self.integral_limit = float(integral_limit)
        self.reset()

    def reset(self):
        self.previous_translation = None
        self.previous_rotation = None
        self.translation_integral = np.zeros(3)
        self.rotation_integral = np.zeros(3)
        self.previous_translation_error = None
        self.previous_rotation_error = None

    @staticmethod
    def _bounded_change(value, previous, maximum_norm):
        if previous is None:
            return value
        change = value - previous
        norm = float(np.linalg.norm(change))
        return previous + change * (maximum_norm / norm) if norm > maximum_norm else value

    @staticmethod
    def _bounded_norm(value, maximum_norm):
        norm = float(np.linalg.norm(value))
        return value * (maximum_norm / norm) if norm > maximum_norm else value

    def next_action(self, current_eef, target_eef, current_quat_xyzw,
                    target_quat_xyzw, reference_action=None) -> np.ndarray:
        translation_error = (
            np.asarray(target_eef, float) - np.asarray(current_eef, float))
        rotation_error = quaternion_error_axis_angle(
            current_quat_xyzw, target_quat_xyzw)
        self.translation_integral = self._bounded_norm(
            self.translation_integral + translation_error, self.integral_limit)
        self.rotation_integral = self._bounded_norm(
            self.rotation_integral + rotation_error, self.integral_limit)
        translation_derivative = (
            np.zeros(3) if self.previous_translation_error is None
            else translation_error - self.previous_translation_error)
        rotation_derivative = (
            np.zeros(3) if self.previous_rotation_error is None
            else rotation_error - self.previous_rotation_error)
        self.previous_translation_error = translation_error.copy()
        self.previous_rotation_error = rotation_error.copy()
        translation = self._bounded_norm(
            translation_error
            + self.translation_integral_gain * self.translation_integral
            + self.translation_derivative_gain * translation_derivative,
            self.maximum_translation_m)
        rotation = self._bounded_norm(
            rotation_error
            + self.rotation_integral_gain * self.rotation_integral
            + self.rotation_derivative_gain * rotation_derivative,
            self.maximum_rotation_rad)
        if float(np.linalg.norm(rotation_error)) <= self.orientation_tolerance_rad:
            rotation = np.zeros(3)
            self.rotation_integral[:] = 0.0
        translation = self._bounded_change(
            translation, self.previous_translation, self.maximum_translation_change_m)
        rotation = self._bounded_change(
            rotation, self.previous_rotation, self.maximum_rotation_change_rad)
        self.previous_translation = translation.copy()
        self.previous_rotation = rotation.copy()
        action = np.zeros(7, dtype=float)
        action[:3] = np.clip(translation / self.translation_action_scale_m, -1, 1)
        action[3:6] = np.clip(rotation / self.rotation_action_scale_rad, -1, 1)
        if reference_action is not None:
            reference = np.asarray(reference_action, float)
            action[6] = reference[6] if len(reference) > 6 else 0.0
        return action


@dataclass(frozen=True)
class CollisionRiskEvidence:
    """Collision-only evidence for one candidate swept motion.

    Values are conservative risk proxies in [0, 1], not calibrated physical
    collision probabilities.  Unknown external occupancy stays explicit and
    can never be silently converted to clear.
    """

    self_collision: float
    fixed_environment: float
    blocked_direction: float
    historical_tube: float
    external_occupancy: str = "unknown"  # clear, blocked, or unknown


@dataclass(frozen=True)
class CollisionRiskDecision:
    risk: float
    safety_score: float
    aggregation: str
    reasons: tuple[str, ...]
    rejected: bool


class CollisionRiskScorer:
    """Conservatively rank collision risk after hard kinematic gates pass."""

    def __init__(self, *, hard_risk: float = 0.9,
                 maximum_aggregate_risk: float = 0.85,
                 unknown_external_floor: float = 0.2,
                 include_historical_tube: bool = True,
                 aggregation: str = "probabilistic_union"):
        if aggregation not in {"probabilistic_union", "maximum"}:
            raise ValueError("unsupported collision-risk aggregation")
        if not 0.0 <= hard_risk <= 1.0:
            raise ValueError("hard_risk must be in [0, 1]")
        if not 0.0 <= maximum_aggregate_risk <= 1.0:
            raise ValueError("maximum_aggregate_risk must be in [0, 1]")
        if not 0.0 <= unknown_external_floor <= 1.0:
            raise ValueError("unknown_external_floor must be in [0, 1]")
        self.hard_risk = float(hard_risk)
        self.maximum_aggregate_risk = float(maximum_aggregate_risk)
        self.unknown_external_floor = float(unknown_external_floor)
        self.include_historical_tube = bool(include_historical_tube)
        self.aggregation = aggregation

    @staticmethod
    def _bounded(value: float, name: str) -> float:
        value = float(value)
        if not np.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be finite and in [0, 1]")
        return value

    def decide(self, evidence: CollisionRiskEvidence) -> CollisionRiskDecision:
        if evidence.external_occupancy not in {"clear", "blocked", "unknown"}:
            raise ValueError("external_occupancy must be clear, blocked, or unknown")
        risks = {
            "self_collision": self._bounded(evidence.self_collision, "self_collision"),
            "fixed_environment": self._bounded(evidence.fixed_environment, "fixed_environment"),
            "blocked_direction": self._bounded(evidence.blocked_direction, "blocked_direction"),
        }
        tube_risk = self._bounded(evidence.historical_tube, "historical_tube")
        if self.include_historical_tube:
            risks["historical_tube"] = tube_risk
        if evidence.external_occupancy == "blocked":
            risks["external_occupancy"] = 1.0
        elif evidence.external_occupancy == "unknown":
            risks["external_occupancy"] = self.unknown_external_floor
        else:
            risks["external_occupancy"] = 0.0

        if self.aggregation == "maximum":
            risk = max(risks.values())
        else:
            safe_product = float(np.prod([1.0 - value for value in risks.values()]))
            risk = 1.0 - safe_product
        reasons = tuple(name for name, value in risks.items() if value >= self.hard_risk)
        if risk >= self.maximum_aggregate_risk:
            reasons = reasons + ("aggregate_collision_risk",)
        return CollisionRiskDecision(
            risk=risk,
            safety_score=1.0 - risk,
            aggregation=self.aggregation,
            reasons=reasons,
            rejected=bool(reasons),
        )


@dataclass(frozen=True)
class LocalEscapeProposal:
    proposal_id: str
    translation_delta: tuple[float, float, float]
    source: str


class LocalEscapeCandidateGenerator:
    """Generate task-agnostic bounded translations without choosing a winner."""

    def __init__(self, step_sizes_m: Sequence[float] = (0.002, 0.0035, 0.005),
                 include_hold: bool = True):
        steps = tuple(float(step) for step in step_sizes_m)
        if not steps or any(not np.isfinite(step) or step <= 0 for step in steps):
            raise ValueError("step sizes must be finite and positive")
        self.step_sizes_m = steps
        self.include_hold = bool(include_hold)

    @staticmethod
    def _unit(vector) -> np.ndarray | None:
        value = np.asarray(vector, dtype=float)
        if value.shape != (3,) or not np.all(np.isfinite(value)):
            return None
        norm = float(np.linalg.norm(value))
        return value / norm if norm > 1e-9 else None

    def generate(self, *, blocked_normal=None, join_direction=None,
                 extra_directions: Iterable[Sequence[float]] = ()) -> tuple[LocalEscapeProposal, ...]:
        directions: list[tuple[str, np.ndarray]] = []
        axes = np.eye(3)
        for axis_index, axis in enumerate(axes):
            directions.extend(((f"axis_{axis_index}_positive", axis),
                               (f"axis_{axis_index}_negative", -axis)))
        away = self._unit(None if blocked_normal is None else -np.asarray(blocked_normal, float))
        if away is not None:
            directions.insert(0, ("away_from_blocked_direction", away))
        join = self._unit(join_direction)
        if join is not None:
            directions.append(("toward_join_state", join))
        for index, direction in enumerate(extra_directions):
            unit = self._unit(direction)
            if unit is not None:
                directions.append((f"extra_{index}", unit))

        unique: list[tuple[str, np.ndarray]] = []
        for source, direction in directions:
            if not any(abs(float(existing @ direction)) > 1.0 - 1e-8
                       and float(existing @ direction) > 0 for _, existing in unique):
                unique.append((source, direction))
        proposals = []
        if self.include_hold:
            proposals.append(LocalEscapeProposal("hold", (0.0, 0.0, 0.0), "hold"))
        for source, direction in unique:
            for step in self.step_sizes_m:
                delta = tuple(float(x) for x in direction * step)
                proposals.append(LocalEscapeProposal(
                    f"{source}_{int(round(step * 1e6))}um", delta, source))
        return tuple(proposals)


@dataclass(frozen=True)
class AdaptiveStepConfig:
    levels_m: tuple[tuple[float, ...], ...] = (
        (0.002, 0.0035, 0.005),
        (0.001, 0.002, 0.003),
        (0.0005, 0.001, 0.002),
    )
    successes_before_expand: int = 3


@dataclass(frozen=True)
class AdaptiveStepDecision:
    action: str
    level: int
    step_sizes_m: tuple[float, ...]
    reason: str


class AdaptiveEscapeStepScheduler:
    """Turn reduce-step requests into bounded candidate scales."""

    def __init__(self, config: AdaptiveStepConfig = AdaptiveStepConfig()):
        if not config.levels_m or config.successes_before_expand < 1:
            raise ValueError("adaptive step schedule must be non-empty")
        previous_max = float("inf")
        for level in config.levels_m:
            if not level or any(step <= 0 or not np.isfinite(step) for step in level):
                raise ValueError("every step level must contain positive finite values")
            if max(level) >= previous_max:
                raise ValueError("step levels must strictly shrink")
            previous_max = max(level)
        self.config = config
        self.reset()

    def reset(self):
        self.level = 0
        self.safe_successes = 0

    @property
    def step_sizes_m(self) -> tuple[float, ...]:
        return self.config.levels_m[self.level]

    def all_candidates_rejected(self) -> AdaptiveStepDecision:
        self.safe_successes = 0
        if self.level + 1 >= len(self.config.levels_m):
            return AdaptiveStepDecision(
                "exhausted", self.level, self.step_sizes_m, "minimum_scale_rejected")
        self.level += 1
        return AdaptiveStepDecision(
            "rescore", self.level, self.step_sizes_m, "reduced_candidate_scale")

    def step_observed(self, *, safe: bool, collision_risk_improved: bool) -> AdaptiveStepDecision:
        if not safe:
            self.safe_successes = 0
            if self.level + 1 < len(self.config.levels_m):
                self.level += 1
                return AdaptiveStepDecision(
                    "rescore", self.level, self.step_sizes_m, "unsafe_response_reduce_scale")
            return AdaptiveStepDecision(
                "exhausted", self.level, self.step_sizes_m, "unsafe_at_minimum_scale")
        self.safe_successes = self.safe_successes + 1 if collision_risk_improved else 0
        if (self.safe_successes >= self.config.successes_before_expand
                and self.level > 0):
            self.level -= 1
            self.safe_successes = 0
            return AdaptiveStepDecision(
                "rescore", self.level, self.step_sizes_m, "sustained_safe_improvement_expand")
        return AdaptiveStepDecision("continue", self.level, self.step_sizes_m, "scale_held")


@dataclass(frozen=True)
class EscapeProgressConfig:
    window: int = 4
    minimum_collision_risk_improvement: float = 0.02
    minimum_join_error_improvement: float = 0.002
    maximum_collision_risk_regression: float = 0.08


@dataclass(frozen=True)
class EscapeProgressDecision:
    state: str
    collision_risk_change: float
    join_error_change: float
    reason: str


class EscapeProgressMonitor:
    """Detect useful multi-step escape progress without rewarding motion alone."""

    def __init__(self, config: EscapeProgressConfig = EscapeProgressConfig()):
        if config.window < 2:
            raise ValueError("escape progress window must be at least two")
        self.config = config
        self.reset()

    def reset(self):
        self.collision_risks = deque(maxlen=self.config.window)
        self.join_errors = deque(maxlen=self.config.window)

    def observe(self, *, collision_risk: float, join_error: float) -> EscapeProgressDecision:
        self.collision_risks.append(float(collision_risk))
        self.join_errors.append(float(join_error))
        if len(self.collision_risks) < self.config.window:
            return EscapeProgressDecision("warming_up", 0.0, 0.0, "insufficient_history")
        risk_change = self.collision_risks[0] - self.collision_risks[-1]
        join_change = self.join_errors[0] - self.join_errors[-1]
        if risk_change < -self.config.maximum_collision_risk_regression:
            return EscapeProgressDecision(
                "replan", risk_change, join_change, "collision_risk_regressed")
        if (risk_change >= self.config.minimum_collision_risk_improvement
                or join_change >= self.config.minimum_join_error_improvement):
            return EscapeProgressDecision("progress", risk_change, join_change, "escape_improved")
        return EscapeProgressDecision("stalled", risk_change, join_change, "no_safety_or_join_progress")


@dataclass(frozen=True)
class EscapeCompletionConfig:
    collision_risk_improvement_for_completion: float = 0.10
    join_error_improvement_for_completion: float = 0.05


class EscapeCompletionEstimator:
    """Monotonic phase evidence from safety margin or historical convergence."""

    def __init__(self, config: EscapeCompletionConfig = EscapeCompletionConfig()):
        if (config.collision_risk_improvement_for_completion <= 0
                or config.join_error_improvement_for_completion <= 0):
            raise ValueError("escape completion spans must be positive")
        self.config = config
        self.reset()

    def reset(self):
        self.baseline_collision_risk = None
        self.baseline_join_error = None
        self.completion = 0.0

    def observe(self, *, collision_risk: float, join_error: float) -> float:
        risk = float(collision_risk)
        error = float(join_error)
        if self.baseline_collision_risk is None:
            self.baseline_collision_risk = risk
            self.baseline_join_error = error
            return self.completion
        risk_progress = (
            self.baseline_collision_risk - risk
        ) / self.config.collision_risk_improvement_for_completion
        join_progress = (
            self.baseline_join_error - error
        ) / self.config.join_error_improvement_for_completion
        observed = float(np.clip(max(risk_progress, join_progress), 0.0, 1.0))
        self.completion = max(self.completion, observed)
        return self.completion


@dataclass(frozen=True)
class SweptCollisionObservation:
    """Raw evidence along the complete candidate sweep, never endpoint-only."""

    minimum_self_clearance_m: float | None
    minimum_fixed_clearance_m: float | None
    clearance_uncertainty_m: float
    blocked_normal: tuple[float, float, float] | None
    predicted_actual_delta: tuple[float, float, float]
    historical_tube_deviation_rad: float | None
    external_occupancy: str = "unknown"
    baseline_self_clearance_m: float | None = None
    final_self_clearance_m: float | None = None
    baseline_fixed_clearance_m: float | None = None
    final_fixed_clearance_m: float | None = None
    baseline_historical_tube_deviation_rad: float | None = None


@dataclass(frozen=True)
class CollisionEvidenceCalibration:
    self_safe_clearance_m: float = 0.03
    fixed_safe_clearance_m: float = 0.04
    clearance_transition_m: float = 0.01
    uncertainty_multiplier: float = 2.0
    tube_safe_deviation_rad: float = 0.05
    tube_transition_rad: float = 0.03
    unknown_geometry_risk: float = 0.35
    blocked_alignment_midpoint: float = 0.0
    blocked_alignment_slope: float = 8.0
    existing_contact_transition_m: float = 0.002


class CollisionEvidenceBuilder:
    """Convert physical clearances and residual direction into risk proxies."""

    def __init__(self, config: CollisionEvidenceCalibration = CollisionEvidenceCalibration()):
        self.config = config

    @staticmethod
    def _sigmoid(value: float) -> float:
        return float(1.0 / (1.0 + np.exp(-np.clip(value, -60.0, 60.0))))

    def _clearance_risk(self, clearance: float | None, safe_clearance: float,
                        uncertainty: float, baseline: float | None = None,
                        final: float | None = None) -> float:
        if clearance is None or not np.isfinite(clearance):
            return float(self.config.unknown_geometry_risk)
        # If contact / sub-margin proximity already exists, alpha=0 belongs to
        # every candidate sweep.  Ranking by the absolute minimum would reject
        # even a separating action.  In that case score whether the candidate
        # worsens or relieves the inherited contact.  New contacts still use
        # the absolute robust clearance branch below.
        if (baseline is not None and final is not None
                and np.isfinite(baseline) and np.isfinite(final)
                and float(baseline) < safe_clearance):
            improvement = float(final) - float(baseline)
            scale = max(float(self.config.existing_contact_transition_m), 1e-9)
            return self._sigmoid(-improvement / scale)
        robust = float(clearance) - self.config.uncertainty_multiplier * max(0.0, float(uncertainty))
        scale = max(float(self.config.clearance_transition_m), 1e-9)
        return self._sigmoid((safe_clearance - robust) / scale)

    def build(self, observation: SweptCollisionObservation) -> CollisionRiskEvidence:
        cfg = self.config
        self_risk = self._clearance_risk(
            observation.minimum_self_clearance_m, cfg.self_safe_clearance_m,
            observation.clearance_uncertainty_m,
            observation.baseline_self_clearance_m,
            observation.final_self_clearance_m)
        fixed_risk = self._clearance_risk(
            observation.minimum_fixed_clearance_m, cfg.fixed_safe_clearance_m,
            observation.clearance_uncertainty_m,
            observation.baseline_fixed_clearance_m,
            observation.final_fixed_clearance_m)

        normal = LocalEscapeCandidateGenerator._unit(observation.blocked_normal)
        delta = LocalEscapeCandidateGenerator._unit(observation.predicted_actual_delta)
        if normal is None or delta is None:
            blocked_risk = 0.0
        else:
            alignment = float(normal @ delta)
            blocked_risk = self._sigmoid(
                cfg.blocked_alignment_slope * (alignment - cfg.blocked_alignment_midpoint))

        deviation = observation.historical_tube_deviation_rad
        if deviation is None or not np.isfinite(deviation):
            tube_risk = float(cfg.unknown_geometry_risk)
        elif (observation.baseline_historical_tube_deviation_rad is not None
              and np.isfinite(observation.baseline_historical_tube_deviation_rad)
              and float(observation.baseline_historical_tube_deviation_rad)
              > cfg.tube_safe_deviation_rad):
            improvement = (
                float(observation.baseline_historical_tube_deviation_rad)
                - float(deviation))
            tube_risk = self._sigmoid(
                -improvement / max(cfg.tube_transition_rad, 1e-9))
        else:
            tube_risk = self._sigmoid(
                (float(deviation) - cfg.tube_safe_deviation_rad)
                / max(cfg.tube_transition_rad, 1e-9))
        return CollisionRiskEvidence(
            self_collision=self_risk,
            fixed_environment=fixed_risk,
            blocked_direction=blocked_risk,
            historical_tube=tube_risk,
            external_occupancy=observation.external_occupancy,
        )


@dataclass(frozen=True)
class HistoricalJoinState:
    action_index: int
    eef_pos: tuple[float, float, float]
    joint_pos: tuple[float, ...]
    eef_rotation: tuple[float, ...] = ()
    joint_velocity: tuple[float, ...] = ()
    gripper_state: float = 0.0


@dataclass(frozen=True)
class JoinErrorConfig:
    joint_weight: float = 0.55
    cartesian_weight: float = 0.30
    velocity_weight: float = 0.10
    gripper_weight: float = 0.05
    orientation_weight: float = 0.15
    error_scale: float = 1.0
    absolute_weight: float = 0.5
    progress_scale: float = 0.005


class HistoricalJoinErrorScorer:
    """Score closeness to compatible replay states independently of safety."""

    def __init__(self, config: JoinErrorConfig = JoinErrorConfig()):
        self.config = config

    def error(self, *, predicted_eef, predicted_joint, predicted_joint_velocity,
              gripper_state: float, historical_state: HistoricalJoinState,
              predicted_eef_rotation=None) -> float:
        cfg = self.config
        joint = float(np.linalg.norm(
            np.asarray(predicted_joint, float) - np.asarray(historical_state.joint_pos, float)))
        cartesian = float(np.linalg.norm(
            np.asarray(predicted_eef, float) - np.asarray(historical_state.eef_pos, float)))
        if historical_state.joint_velocity and predicted_joint_velocity is not None:
            velocity = float(np.linalg.norm(
                np.asarray(predicted_joint_velocity, float)
                - np.asarray(historical_state.joint_velocity, float)))
        else:
            velocity = 0.0
        gripper = abs(float(gripper_state) - float(historical_state.gripper_state))
        if historical_state.eef_rotation and predicted_eef_rotation is not None:
            orientation = float(np.linalg.norm(quaternion_error_axis_angle(
                predicted_eef_rotation, historical_state.eef_rotation)))
        else:
            orientation = 0.0
        return (
            cfg.joint_weight * joint
            + cfg.cartesian_weight * cartesian
            + cfg.velocity_weight * velocity
            + cfg.gripper_weight * gripper
            + cfg.orientation_weight * orientation
        )

    def score(self, **kwargs) -> float:
        error = self.error(**kwargs)
        scale = max(float(self.config.error_scale), 1e-12)
        return float(np.exp(-0.5 * (error / scale) ** 2))

    def score_with_progress(self, *, current_error: float, **kwargs) -> float:
        predicted_error = self.error(**kwargs)
        absolute = float(np.exp(
            -0.5 * (predicted_error / max(float(self.config.error_scale), 1e-12)) ** 2))
        progress = 1.0 / (1.0 + np.exp(
            -(float(current_error) - predicted_error)
            / max(float(self.config.progress_scale), 1e-12)))
        weight = float(np.clip(self.config.absolute_weight, 0.0, 1.0))
        return weight * absolute + (1.0 - weight) * float(progress)


@dataclass(frozen=True)
class EscapeCandidate:
    candidate_id: str
    action: tuple[float, ...]
    predicted_eef: tuple[float, float, float]
    predicted_joint: tuple[float, ...]
    predicted_joint_velocity: tuple[float, ...]
    collision_evidence: CollisionRiskEvidence
    model_uncertainty: float = 0.0
    predicted_eef_rotation: tuple[float, ...] = ()


@dataclass(frozen=True)
class EscapeCandidateScore:
    candidate_id: str
    collision_safety: float
    join_score: float
    uncertainty: float
    join_weight: float
    total_score: float
    rejected: bool
    reasons: tuple[str, ...]


class DynamicJoinWeight:
    """Logistic shift from collision avoidance toward historical joining."""

    def __init__(self, minimum: float = 0.1, maximum: float = 0.8,
                 slope: float = 10.0, midpoint: float = 0.6):
        if not 0.0 <= minimum <= maximum <= 1.0:
            raise ValueError("join-weight bounds must lie in [0, 1]")
        self.minimum = float(minimum)
        self.maximum = float(maximum)
        self.slope = float(slope)
        self.midpoint = float(midpoint)

    def __call__(self, escape_completion: float) -> float:
        z = float(np.clip(escape_completion, 0.0, 1.0))
        logistic = 1.0 / (1.0 + np.exp(-self.slope * (z - self.midpoint)))
        return self.minimum + (self.maximum - self.minimum) * float(logistic)


class EscapeCandidateEvaluator:
    """Rank hard-gate-approved local actions without conflating safety and error."""

    def __init__(self, collision_scorer: CollisionRiskScorer | None = None,
                 join_scorer: HistoricalJoinErrorScorer | None = None,
                 join_weight: DynamicJoinWeight | None = None,
                 maximum_uncertainty: float = 1.0,
                 uncertainty_penalty: float = 0.2):
        self.collision_scorer = collision_scorer or CollisionRiskScorer()
        self.join_scorer = join_scorer or HistoricalJoinErrorScorer()
        self.join_weight = join_weight or DynamicJoinWeight()
        self.maximum_uncertainty = float(maximum_uncertainty)
        self.uncertainty_penalty = float(uncertainty_penalty)

    def evaluate(self, candidate: EscapeCandidate, *, historical_state: HistoricalJoinState,
                 gripper_state: float, escape_completion: float,
                 current_eef=None, current_joint=None,
                 current_joint_velocity=None,
                 current_eef_rotation=None) -> EscapeCandidateScore:
        collision = self.collision_scorer.decide(candidate.collision_evidence)
        uncertainty = float(candidate.model_uncertainty)
        reasons = list(collision.reasons)
        if not np.isfinite(uncertainty) or uncertainty > self.maximum_uncertainty:
            reasons.append("prediction_uncertainty")
        weight = self.join_weight(escape_completion)
        join_kwargs = dict(
            predicted_eef=candidate.predicted_eef,
            predicted_joint=candidate.predicted_joint,
            predicted_joint_velocity=candidate.predicted_joint_velocity,
            gripper_state=gripper_state,
            historical_state=historical_state,
            predicted_eef_rotation=(candidate.predicted_eef_rotation or None),
        )
        if current_eef is not None and current_joint is not None:
            current_error = self.join_scorer.error(
                predicted_eef=current_eef,
                predicted_joint=current_joint,
                predicted_joint_velocity=current_joint_velocity,
                gripper_state=gripper_state,
                historical_state=historical_state,
                predicted_eef_rotation=current_eef_rotation,
            )
            join = self.join_scorer.score_with_progress(
                current_error=current_error, **join_kwargs)
        else:
            join = self.join_scorer.score(**join_kwargs)
        total = (
            weight * join
            + (1.0 - weight) * collision.safety_score
            - self.uncertainty_penalty * max(0.0, uncertainty)
        )
        return EscapeCandidateScore(
            candidate_id=candidate.candidate_id,
            collision_safety=collision.safety_score,
            join_score=join,
            uncertainty=uncertainty,
            join_weight=weight,
            total_score=float(total),
            rejected=collision.rejected or "prediction_uncertainty" in reasons,
            reasons=tuple(reasons),
        )

    def select(self, candidates: Iterable[EscapeCandidate], **kwargs) -> EscapeCandidateScore | None:
        scores = [self.evaluate(candidate, **kwargs) for candidate in candidates]
        viable = [score for score in scores if not score.rejected]
        return max(viable, key=lambda score: score.total_score) if viable else None


@dataclass(frozen=True)
class ReverseReplayReference:
    action_index: int
    joint_target: tuple[float, ...]
    eef_target: tuple[float, float, float]
    eef_rotation_target: tuple[float, ...] = ()


class ReverseStateReplayPlanner:
    """Replay dense historical states backwards; never negate old actions."""

    def build(self, history: Iterable[Mapping], *, join_action_index: int,
              target_action_index: int) -> tuple[ReverseReplayReference, ...]:
        if target_action_index > join_action_index:
            raise ValueError("target must precede or equal join state")
        rows = sorted(
            (row for row in history
             if target_action_index <= int(row["action_index"]) <= join_action_index
             and bool(row.get("recovery_safe", False))),
            key=lambda row: int(row["action_index"]), reverse=True)
        return tuple(ReverseReplayReference(
            action_index=int(row["action_index"]),
            joint_target=tuple(float(x) for x in row["joint_pos"]),
            eef_target=tuple(float(x) for x in row["eef_pos"]),
            eef_rotation_target=tuple(float(x) for x in row.get(
                "eef_rotation", row.get("eef_quat", ()))),
        ) for row in rows)


@dataclass(frozen=True)
class JoinReadinessConfig:
    maximum_joint_error_rad: float = 0.08
    maximum_eef_error_m: float = 0.01
    maximum_joint_speed_rad_s: float = 0.12
    maximum_orientation_error_rad: float = 0.10
    minimum_collision_safety: float = 0.5
    confirmations: int = 3
    require_joint_match: bool = True


class JoinReadinessGate:
    """Require a stable, safe state before reverse-state replay is allowed."""

    def __init__(self, config: JoinReadinessConfig = JoinReadinessConfig()):
        if config.confirmations < 1:
            raise ValueError("join confirmations must be positive")
        self.config = config
        self.reset()

    def reset(self):
        self.consecutive = 0

    def observe(self, *, joint_error_rad: float, eef_error_m: float,
                maximum_joint_speed_rad_s: float,
                collision_safety: float, hard_gate_passed: bool,
                orientation_error_rad: float = 0.0) -> bool:
        cfg = self.config
        ready_now = (
            hard_gate_passed
            and (not cfg.require_joint_match
                 or float(joint_error_rad) <= cfg.maximum_joint_error_rad)
            and float(eef_error_m) <= cfg.maximum_eef_error_m
            and float(maximum_joint_speed_rad_s) <= cfg.maximum_joint_speed_rad_s
            and float(orientation_error_rad) <= cfg.maximum_orientation_error_rad
            and float(collision_safety) >= cfg.minimum_collision_safety
        )
        self.consecutive = self.consecutive + 1 if ready_now else 0
        return self.consecutive >= cfg.confirmations


@dataclass(frozen=True)
class ReverseReplayCursorDecision:
    action: str
    reference: ReverseReplayReference | None
    cursor: int
    reason: str


class ReverseReplayCursor:
    """Advance dense replay references only after the current state is reached."""

    def __init__(self, references: Sequence[ReverseReplayReference],
                 maximum_steps_per_reference: int = 20):
        if maximum_steps_per_reference < 1:
            raise ValueError("maximum_steps_per_reference must be positive")
        self.references = tuple(references)
        self.maximum_steps_per_reference = int(maximum_steps_per_reference)
        self.reset()

    def reset(self):
        self.cursor = 0
        self.steps_on_reference = 0
        self.failed = False

    def current(self) -> ReverseReplayReference | None:
        return self.references[self.cursor] if self.cursor < len(self.references) else None

    def observe(self, *, reached: bool, monitor_state: str) -> ReverseReplayCursorDecision:
        if self.failed:
            return ReverseReplayCursorDecision("abort", self.current(), self.cursor, "cursor_failed")
        if monitor_state == "safe_stop":
            self.failed = True
            return ReverseReplayCursorDecision("safe_stop", self.current(), self.cursor, "hard_fault")
        if monitor_state not in {"safe", "warming_up"}:
            self.failed = True
            return ReverseReplayCursorDecision("rejoin", self.current(), self.cursor, "replay_monitor")
        if self.current() is None:
            return ReverseReplayCursorDecision("complete", None, self.cursor, "all_references_reached")
        if reached:
            self.cursor += 1
            self.steps_on_reference = 0
            if self.current() is None:
                return ReverseReplayCursorDecision("complete", None, self.cursor, "all_references_reached")
            return ReverseReplayCursorDecision("track", self.current(), self.cursor, "next_reference")
        self.steps_on_reference += 1
        if self.steps_on_reference >= self.maximum_steps_per_reference:
            self.failed = True
            return ReverseReplayCursorDecision(
                "rejoin", self.current(), self.cursor, "reference_timeout")
        return ReverseReplayCursorDecision("track", self.current(), self.cursor, "continue_reference")


@dataclass(frozen=True)
class HybridRecoveryConfig:
    settle_confirmations: int = 3
    maximum_no_candidate_cycles: int = 3
    maximum_rejoin_attempts: int = 3
    maximum_escape_steps: int = 100


@dataclass(frozen=True)
class HybridRecoveryDecision:
    state: str
    action: str
    reason: str
    rejoin_attempts: int
    no_candidate_cycles: int


class HybridRecoveryStateMachine:
    """Control-plane state machine; it never executes a robot action itself."""

    VALID_STATES = {
        "idle", "hold_and_settle", "safe_escape", "reverse_replay",
        "reobserve_and_replan", "safe_stop",
    }

    def __init__(self, config: HybridRecoveryConfig = HybridRecoveryConfig()):
        if min(config.settle_confirmations, config.maximum_no_candidate_cycles,
               config.maximum_rejoin_attempts, config.maximum_escape_steps) < 1:
            raise ValueError("hybrid recovery budgets must be positive")
        self.config = config
        self.reset()

    def reset(self):
        self.state = "idle"
        self.settle_run = 0
        self.no_candidate_cycles = 0
        self.rejoin_attempts = 0
        self.escape_steps = 0

    def _decision(self, action: str, reason: str) -> HybridRecoveryDecision:
        return HybridRecoveryDecision(
            self.state, action, reason, self.rejoin_attempts, self.no_candidate_cycles)

    def trigger(self) -> HybridRecoveryDecision:
        if self.state != "idle":
            return self._decision("hold", "recovery_already_active")
        self.state = "hold_and_settle"
        return self._decision("hold", "recovery_triggered")

    def hard_fault(self, reason: str = "hard_fault") -> HybridRecoveryDecision:
        self.state = "safe_stop"
        return self._decision("safe_stop", reason)

    def observe_settle(self, *, stable: bool, hard_fault: bool = False) -> HybridRecoveryDecision:
        if hard_fault:
            return self.hard_fault("hard_fault_during_settle")
        if self.state != "hold_and_settle":
            return self._decision("hold", "settle_observation_outside_state")
        self.settle_run = self.settle_run + 1 if stable else 0
        if self.settle_run >= self.config.settle_confirmations:
            self.state = "safe_escape"
            self.settle_run = 0
            return self._decision("score_escape_candidates", "settled")
        return self._decision("hold", "waiting_for_stability")

    def candidates_scored(self, *, selected: bool,
                          hard_fault: bool = False) -> HybridRecoveryDecision:
        if hard_fault:
            return self.hard_fault("hard_fault_during_candidate_scoring")
        if self.state != "safe_escape":
            return self._decision("hold", "candidate_result_outside_escape")
        if not selected:
            self.no_candidate_cycles += 1
            if self.no_candidate_cycles >= self.config.maximum_no_candidate_cycles:
                return self.hard_fault("no_viable_escape_candidate")
            return self._decision("reduce_step_and_rescore", "all_candidates_rejected")
        self.no_candidate_cycles = 0
        self.escape_steps += 1
        if self.escape_steps > self.config.maximum_escape_steps:
            return self.hard_fault("escape_step_budget_exhausted")
        return self._decision("execute_escape_step", "candidate_selected")

    def escape_observed(self, *, response_state: str,
                        join_ready: bool) -> HybridRecoveryDecision:
        if self.state != "safe_escape":
            return self._decision("hold", "escape_observation_outside_state")
        if response_state == "safe_stop":
            return self.hard_fault("hard_fault_after_escape_step")
        if response_state not in {"safe", "warming_up"}:
            self.rejoin_attempts += 1
            if self.rejoin_attempts >= self.config.maximum_rejoin_attempts:
                return self.hard_fault("escape_response_retry_exhausted")
            self.state = "hold_and_settle"
            self.settle_run = 0
            return self._decision("hold", "escape_response_rejected")
        if join_ready:
            self.state = "reverse_replay"
            return self._decision("start_reverse_replay", "join_confirmed")
        return self._decision("score_escape_candidates", "continue_escape")

    def replay_observed(self, *, replay_action: str) -> HybridRecoveryDecision:
        if self.state != "reverse_replay":
            return self._decision("hold", "replay_observation_outside_state")
        if replay_action == "safe_stop":
            return self.hard_fault("hard_fault_during_replay")
        if replay_action in {"rejoin", "abort"}:
            self.rejoin_attempts += 1
            if self.rejoin_attempts >= self.config.maximum_rejoin_attempts:
                return self.hard_fault("replay_rejoin_budget_exhausted")
            self.state = "hold_and_settle"
            self.settle_run = 0
            return self._decision("hold", "replay_interrupted")
        if replay_action == "complete":
            self.state = "reobserve_and_replan"
            return self._decision("reobserve_world", "rollback_target_reached")
        return self._decision("track_reverse_reference", "replay_continues")


@dataclass(frozen=True)
class EscapeCoordinatorDecision:
    """Auditable control-plane result; execution remains the caller's job."""

    state: str
    action: str
    reason: str
    step_level: int
    step_sizes_m: tuple[float, ...]
    candidate_id: str | None = None
    candidate_action: tuple[float, ...] | None = None
    candidate_score: float | None = None
    response_state: str | None = None
    progress_state: str | None = None
    join_ready: bool = False
    replay_reference_action_index: int | None = None


class HybridEscapeCoordinator:
    """Compose escape scoring, adaptive scale and temporal safety evidence.

    The coordinator deliberately owns no simulator or robot handle.  A caller
    must rebuild physical evidence and candidates after every executed step.
    """

    def __init__(self, *,
                 machine: HybridRecoveryStateMachine | None = None,
                 scheduler: AdaptiveEscapeStepScheduler | None = None,
                 evaluator: EscapeCandidateEvaluator | None = None,
                 response_monitor: WindowedRecoveryResponseMonitor | None = None,
                 progress_monitor: EscapeProgressMonitor | None = None,
                 completion_estimator: EscapeCompletionEstimator | None = None,
                 join_gate: JoinReadinessGate | None = None,
                 maximum_tolerated_conflict_risk: float = 0.65):
        self.machine = machine or HybridRecoveryStateMachine()
        self.scheduler = scheduler or AdaptiveEscapeStepScheduler()
        self.evaluator = evaluator or EscapeCandidateEvaluator()
        self.response_monitor = response_monitor or WindowedRecoveryResponseMonitor(window=3)
        self.progress_monitor = progress_monitor or EscapeProgressMonitor()
        self.completion_estimator = completion_estimator or EscapeCompletionEstimator()
        self.join_gate = join_gate or JoinReadinessGate()
        if not 0.0 <= maximum_tolerated_conflict_risk <= 1.0:
            raise ValueError("maximum_tolerated_conflict_risk must be in [0, 1]")
        self.maximum_tolerated_conflict_risk = float(maximum_tolerated_conflict_risk)
        self.last_selected_score: EscapeCandidateScore | None = None
        self.previous_collision_risk: float | None = None
        self.replay_cursor: ReverseReplayCursor | None = None

    def _decision(self, machine_decision: HybridRecoveryDecision, *,
                  reason: str | None = None, candidate: EscapeCandidate | None = None,
                  score: EscapeCandidateScore | None = None,
                  response_state: str | None = None,
                  progress_state: str | None = None,
                  join_ready: bool = False,
                  replay_reference: ReverseReplayReference | None = None
                  ) -> EscapeCoordinatorDecision:
        return EscapeCoordinatorDecision(
            state=machine_decision.state,
            action=machine_decision.action,
            reason=reason or machine_decision.reason,
            step_level=self.scheduler.level,
            step_sizes_m=self.scheduler.step_sizes_m,
            candidate_id=candidate.candidate_id if candidate is not None else None,
            candidate_action=candidate.action if candidate is not None else None,
            candidate_score=score.total_score if score is not None else None,
            response_state=response_state,
            progress_state=progress_state,
            join_ready=join_ready,
            replay_reference_action_index=(
                replay_reference.action_index if replay_reference is not None else None),
        )

    def trigger(self) -> EscapeCoordinatorDecision:
        self.scheduler.reset()
        self.response_monitor.reset()
        self.progress_monitor.reset()
        self.completion_estimator.reset()
        self.join_gate.reset()
        self.last_selected_score = None
        self.previous_collision_risk = None
        self.replay_cursor = None
        return self._decision(self.machine.trigger())

    def configure_replay(self, references: Sequence[ReverseReplayReference], *,
                         maximum_steps_per_reference: int = 20):
        if not references:
            raise ValueError("reverse replay requires at least one reference")
        self.replay_cursor = ReverseReplayCursor(
            references, maximum_steps_per_reference=maximum_steps_per_reference)

    def observe_settle(self, *, stable: bool,
                       hard_fault: bool = False) -> EscapeCoordinatorDecision:
        decision = self.machine.observe_settle(stable=stable, hard_fault=hard_fault)
        return self._decision(decision)

    def score_candidates(self, candidates: Iterable[EscapeCandidate], **evaluation_kwargs
                         ) -> EscapeCoordinatorDecision:
        candidate_rows = tuple(candidates)
        evaluation_kwargs.setdefault(
            "escape_completion", self.completion_estimator.completion)
        score = self.evaluator.select(candidate_rows, **evaluation_kwargs)
        if score is None:
            scale = self.scheduler.all_candidates_rejected()
            machine_decision = self.machine.candidates_scored(selected=False)
            if scale.action == "exhausted" and machine_decision.state != "safe_stop":
                machine_decision = self.machine.hard_fault("minimum_scale_rejected")
            return self._decision(machine_decision, reason=scale.reason)
        candidate = next(row for row in candidate_rows if row.candidate_id == score.candidate_id)
        self.last_selected_score = score
        machine_decision = self.machine.candidates_scored(selected=True)
        return self._decision(machine_decision, candidate=candidate, score=score)

    def observe_execution(self, *, commanded_delta, actual_delta,
                          collision_risk: float, join_error: float,
                          joint_error_rad: float, eef_error_m: float,
                          maximum_joint_speed_rad_s: float,
                          collision_safety: float, hard_gate_passed: bool,
                          orientation_error_rad: float = 0.0,
                          commanded_rotation_delta=None,
                          actual_rotation_delta=None,
                          contact_probability: float = 0.0,
                          contact_confidence: float = 0.0,
                          hard_fault: bool = False) -> EscapeCoordinatorDecision:
        response = self.response_monitor.observe(
            commanded_delta=commanded_delta, actual_delta=actual_delta,
            commanded_rotation_delta=commanded_rotation_delta,
            actual_rotation_delta=actual_rotation_delta,
            contact_probability=contact_probability,
            contact_confidence=contact_confidence, hard_fault=hard_fault)
        progress = self.progress_monitor.observe(
            collision_risk=collision_risk, join_error=join_error)
        self.completion_estimator.observe(
            collision_risk=collision_risk, join_error=join_error)
        collision_improved = (
            self.previous_collision_risk is not None
            and float(collision_risk) < self.previous_collision_risk
        )
        self.previous_collision_risk = float(collision_risk)

        effective_response = response.state
        effective_progress_state = progress.state
        risk_conflict_tolerated = (
            response.state == "safe"
            and progress.state == "replan"
            and progress.join_error_change >= self.progress_monitor.config.minimum_join_error_improvement
            and float(collision_risk) <= self.maximum_tolerated_conflict_risk
            and hard_gate_passed
        )
        if risk_conflict_tolerated:
            # Signed-distance contact evidence can switch discontinuously near
            # a contact boundary.  Preserve hard collision gates, but do not
            # call a physically responsive, converging step a failed response
            # solely because the soft risk proxy changed branch.
            effective_progress_state = "progress_with_risk_conflict"
        elif (response.state in {"safe", "warming_up"}
              and progress.state in {"replan", "stalled"}):
            effective_response = "replan"
        join_ready = False
        if response.state == "safe" and progress.state in {"progress", "warming_up"}:
            join_ready = self.join_gate.observe(
                joint_error_rad=joint_error_rad, eef_error_m=eef_error_m,
                maximum_joint_speed_rad_s=maximum_joint_speed_rad_s,
                collision_safety=collision_safety,
                hard_gate_passed=hard_gate_passed,
                orientation_error_rad=orientation_error_rad)
        else:
            self.join_gate.reset()

        scale_decision = None
        if response.state not in {"safe", "warming_up"}:
            scale_decision = self.scheduler.step_observed(
                safe=False, collision_risk_improved=False)
        elif response.state == "safe":
            scale_decision = self.scheduler.step_observed(
                safe=effective_response == "safe",
                collision_risk_improved=collision_improved)

        if join_ready and self.replay_cursor is None:
            machine_decision = self.machine.hard_fault("missing_reverse_replay_plan")
        else:
            machine_decision = self.machine.escape_observed(
                response_state=effective_response, join_ready=join_ready)
        if scale_decision is not None and scale_decision.action == "exhausted":
            machine_decision = self.machine.hard_fault(scale_decision.reason)
        if machine_decision.state == "hold_and_settle":
            self.response_monitor.reset()
            self.progress_monitor.reset()
            self.join_gate.reset()
            self.previous_collision_risk = None
        return self._decision(
            machine_decision, response_state=response.state,
            progress_state=effective_progress_state, join_ready=join_ready)

    def current_replay_reference(self) -> ReverseReplayReference | None:
        return self.replay_cursor.current() if self.replay_cursor is not None else None

    def observe_replay(self, *, reached: bool,
                       monitor_state: str) -> EscapeCoordinatorDecision:
        if self.machine.state != "reverse_replay":
            return self._decision(
                self.machine._decision("hold", "replay_observation_outside_state"))
        if self.replay_cursor is None:
            return self._decision(self.machine.hard_fault("missing_reverse_replay_plan"))
        cursor_decision = self.replay_cursor.observe(
            reached=reached, monitor_state=monitor_state)
        machine_decision = self.machine.replay_observed(
            replay_action=cursor_decision.action)
        return self._decision(
            machine_decision, reason=cursor_decision.reason,
            replay_reference=cursor_decision.reference)


@dataclass(frozen=True)
class RecoveryEscalationConfig:
    maximum_candidate_retries_per_level: int = 2
    maximum_rollback_levels: int = 3


@dataclass(frozen=True)
class RecoveryEscalationDecision:
    action: str
    rollback_level: int
    candidate_retry: int


class RecoveryEscalationController:
    """Bound resampling and progressively choose earlier recovery checkpoints."""

    def __init__(self, config: RecoveryEscalationConfig = RecoveryEscalationConfig()):
        if config.maximum_candidate_retries_per_level < 1 or config.maximum_rollback_levels < 1:
            raise ValueError("recovery escalation budgets must be positive")
        self.config = config
        self.reset()

    def reset(self):
        self.rollback_level = 0
        self.candidate_retry = 0
        self.stopped = False

    def candidate_evaluated(self, *, accepted: bool) -> RecoveryEscalationDecision:
        if self.stopped:
            return RecoveryEscalationDecision(
                "safe_stop", self.rollback_level, self.candidate_retry)
        if accepted:
            self.candidate_retry = 0
            return RecoveryEscalationDecision(
                "execute_short_candidate", self.rollback_level, 0)
        self.candidate_retry += 1
        if self.candidate_retry < self.config.maximum_candidate_retries_per_level:
            return RecoveryEscalationDecision(
                "resample_candidate", self.rollback_level, self.candidate_retry)
        self.candidate_retry = 0
        self.rollback_level += 1
        if self.rollback_level < self.config.maximum_rollback_levels:
            return RecoveryEscalationDecision(
                "rollback_deeper", self.rollback_level, 0)
        self.stopped = True
        return RecoveryEscalationDecision("safe_stop", self.rollback_level, 0)


@dataclass(frozen=True)
class PlanNoveltyDecision:
    reject: bool
    mean_direction_cosine: float
    final_path_separation_m: float
    reason: str


class FailedPlanNoveltyGate:
    """Reject only repeated failure structure, not merely similar actions."""

    def __init__(self, cosine_threshold: float = 0.95,
                 maximum_path_separation_m: float = 0.01,
                 minimum_recovery_progress: float = 0.02,
                 action_scale_m: float = 0.05):
        self.cosine_threshold = float(cosine_threshold)
        self.maximum_path_separation_m = float(maximum_path_separation_m)
        self.minimum_recovery_progress = float(minimum_recovery_progress)
        self.action_scale_m = float(action_scale_m)

    def decide(self, failed_actions, candidate_actions, *, reenters_failure_region: bool,
               predicted_recovery_progress: float) -> PlanNoveltyDecision:
        old = np.asarray(failed_actions, dtype=float)
        new = np.asarray(candidate_actions, dtype=float)
        length = min(len(old), len(new))
        if length == 0:
            return PlanNoveltyDecision(False, 0.0, float("inf"), "insufficient_overlap")
        old = old[:length, :3]
        new = new[:length, :3]
        numerator = np.sum(old * new, axis=1)
        denominator = np.linalg.norm(old, axis=1) * np.linalg.norm(new, axis=1)
        valid = denominator > 1e-9
        cosine = float(np.mean(numerator[valid] / denominator[valid])) if np.any(valid) else 0.0
        old_path = np.cumsum(self.action_scale_m * np.clip(old, -1, 1), axis=0)
        new_path = np.cumsum(self.action_scale_m * np.clip(new, -1, 1), axis=0)
        separation = float(np.linalg.norm(old_path[-1] - new_path[-1]))
        reject = (
            reenters_failure_region
            and cosine >= self.cosine_threshold
            and separation <= self.maximum_path_separation_m
            and predicted_recovery_progress < self.minimum_recovery_progress
        )
        return PlanNoveltyDecision(
            reject, cosine, separation,
            "repeats_failed_local_structure_without_progress" if reject else "candidate_not_proven_repetitive",
        )


@dataclass(frozen=True)
class ReplanLoopDecision:
    state: str
    action: str
    reason: str
    rollback_level: int
    retry_count: int
    novelty: PlanNoveltyDecision | None = None


class RecoveryReplanCoordinator:
    """Bound candidate review, short execution, resampling and deeper rollback.

    The coordinator does not infer actions or predict task success.  Callers
    provide causal evidence and retain responsibility for per-microstep safety
    gates.  Similarity alone never rejects a plan; the existing novelty gate
    also requires re-entry into a known failure region and low progress.
    """

    def __init__(self, novelty_gate: FailedPlanNoveltyGate | None = None,
                 escalation: RecoveryEscalationController | None = None):
        self.novelty_gate = novelty_gate or FailedPlanNoveltyGate()
        self.escalation = escalation or RecoveryEscalationController()
        self.reset()

    def reset(self):
        self.escalation.reset()
        self.state = "ready_for_candidate"

    def _decision(self, action: str, reason: str,
                  novelty: PlanNoveltyDecision | None = None) -> ReplanLoopDecision:
        return ReplanLoopDecision(
            state=self.state, action=action, reason=reason,
            rollback_level=self.escalation.rollback_level,
            retry_count=self.escalation.candidate_retry,
            novelty=novelty)

    def _failed_attempt(self, reason: str,
                        novelty: PlanNoveltyDecision | None = None) -> ReplanLoopDecision:
        escalation = self.escalation.candidate_evaluated(accepted=False)
        if escalation.action == "resample_candidate":
            self.state = "ready_for_candidate"
        elif escalation.action == "rollback_deeper":
            self.state = "awaiting_rollback"
        else:
            self.state = "safe_stop"
        return self._decision(escalation.action, reason, novelty)

    def assess_candidate(self, failed_actions, candidate_actions, *,
                         reenters_failure_region: bool,
                         predicted_recovery_progress: float) -> ReplanLoopDecision:
        if self.state != "ready_for_candidate":
            return self._decision("hold", "candidate_outside_ready_state")
        novelty = self.novelty_gate.decide(
            failed_actions, candidate_actions,
            reenters_failure_region=reenters_failure_region,
            predicted_recovery_progress=predicted_recovery_progress)
        if novelty.reject:
            return self._failed_attempt(novelty.reason, novelty)
        self.state = "executing_short_candidate"
        return self._decision(
            "execute_short_candidate", novelty.reason, novelty)

    def observe_short_segment(self, *, task_success: bool,
                              monitor_state: str,
                              useful_progress: bool) -> ReplanLoopDecision:
        if self.state != "executing_short_candidate":
            return self._decision("hold", "segment_outside_execution_state")
        if task_success:
            self.escalation.candidate_retry = 0
            self.state = "complete"
            return self._decision("complete", "task_success")
        if monitor_state == "safe_stop":
            self.escalation.stopped = True
            self.state = "safe_stop"
            return self._decision("safe_stop", "hard_fault_during_short_segment")
        if monitor_state not in {"safe", "warming_up"}:
            return self._failed_attempt("short_segment_response_rejected")
        if not useful_progress:
            return self._failed_attempt("short_segment_no_useful_progress")
        self.escalation.candidate_retry = 0
        self.state = "ready_for_candidate"
        return self._decision(
            "reobserve_and_replan", "short_segment_made_progress")

    def rollback_completed(self, *, safe: bool) -> ReplanLoopDecision:
        if self.state != "awaiting_rollback":
            return self._decision("hold", "rollback_outside_expected_state")
        if not safe:
            self.escalation.stopped = True
            self.state = "safe_stop"
            return self._decision("safe_stop", "rollback_failed_safety_check")
        self.state = "ready_for_candidate"
        return self._decision("reobserve_and_replan", "deeper_rollback_complete")
