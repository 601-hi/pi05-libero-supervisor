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
    "goal_relation_progress_stalled": {"transport", "place", "goal_relation"},
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
                 corridor_deviation: float, occupancy_clear: bool) -> RecoverySafetyDecision:
        cfg = self.config
        reasons = []
        if np.linalg.norm(np.asarray(predicted_eef) - np.asarray(current_eef)) > cfg.maximum_cartesian_step_m:
            reasons.append("cartesian_step_limit")
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

    def observe(self, *, commanded_delta, actual_delta, contact_probability: float,
                contact_confidence: float, hard_fault: bool = False) -> RecoverySafetyDecision:
        immediate = self.gate.postcheck(
            commanded_delta=[0, 0, 0], actual_delta=[0, 0, 0],
            contact_probability=contact_probability,
            contact_confidence=contact_confidence, hard_fault=hard_fault)
        if immediate.state == "safe_stop":
            return immediate
        self.commands.append(np.asarray(commanded_delta, dtype=float))
        self.actuals.append(np.asarray(actual_delta, dtype=float))
        if len(self.commands) < self.window:
            return RecoverySafetyDecision("warming_up", ())
        return self.gate.postcheck(
            commanded_delta=np.sum(self.commands, axis=0),
            actual_delta=np.sum(self.actuals, axis=0),
            contact_probability=contact_probability,
            contact_confidence=contact_confidence, hard_fault=False)

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
