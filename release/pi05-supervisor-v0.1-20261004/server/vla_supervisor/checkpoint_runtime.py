from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .checkpoint_recovery import (
    CheckpointBuffer, CheckpointSelector, HistoricalJoinState,
    RecoveryCheckpoint, ReverseReplayReference, ReverseStateReplayPlanner,
)


@dataclass(frozen=True)
class CheckpointEvidence:
    phase: str | None
    phase_confidence: float | None
    observability: float | None
    observability_source: str | None
    response_reliability: float | None
    response_source: str | None
    contact_clear: bool | None
    contact_source: str | None
    minimum_self_clearance_m: float | None = None
    minimum_environment_clearance_m: float | None = None
    self_clearance_pair: tuple[str, str] | None = None
    environment_clearance_pair: tuple[str, str] | None = None
    task_progress: float | None = None


@dataclass(frozen=True)
class CheckpointAdmissionDecision:
    admitted: bool
    reasons: tuple[str, ...]
    checkpoint: RecoveryCheckpoint | None
    history_row: Mapping


class OnlineCheckpointRecorder:
    """Fail-closed bridge from online evidence to recovery history.

    Missing evidence is never converted to a favourable default.  The history
    row is retained for audit, while ``recovery_safe`` becomes true only when
    the complete deployable evidence contract and ``CheckpointBuffer`` gates
    both pass.
    """

    def __init__(self, buffer: CheckpointBuffer | None = None,
                 minimum_phase_confidence: float = 0.6):
        self.buffer = buffer or CheckpointBuffer()
        self.minimum_phase_confidence = float(minimum_phase_confidence)
        self.history: list[Mapping] = []

    def reset(self) -> None:
        self.buffer.reset()
        self.history.clear()

    @staticmethod
    def _vector(state: Mapping, key: str, size: int | None = None):
        if key not in state:
            return None
        value = np.asarray(state[key], dtype=float)
        if value.ndim != 1 or (size is not None and len(value) != size):
            return None
        if not np.isfinite(value).all():
            return None
        return tuple(float(x) for x in value)

    def consider(self, *, action_index: int, state: Mapping,
                 evidence: CheckpointEvidence) -> CheckpointAdmissionDecision:
        reasons = []
        eef_pos = self._vector(state, "eef_pos", 3)
        eef_rotation = self._vector(state, "eef_rotation", 4)
        joint_pos = self._vector(state, "joint_pos")
        joint_velocity = self._vector(state, "joint_velocity")
        if joint_velocity is None:
            joint_velocity = self._vector(state, "joint_vel")
        joint_margin = state.get("joint_limit_margin_rad", state.get("joint_margin"))
        singularity = state.get(
            "jacobian_min_singular_value", state.get("singularity_sigma"))
        if eef_pos is None: reasons.append("missing_or_invalid_eef_pos")
        if eef_rotation is None: reasons.append("missing_or_invalid_eef_rotation")
        if joint_pos is None: reasons.append("missing_or_invalid_joint_pos")
        if joint_velocity is None: reasons.append("missing_or_invalid_joint_velocity")
        if joint_margin is None: reasons.append("missing_joint_margin")
        if singularity is None: reasons.append("missing_singularity_margin")
        if evidence.phase is None or evidence.phase_confidence is None:
            reasons.append("missing_phase_evidence")
        elif evidence.phase_confidence < self.minimum_phase_confidence:
            reasons.append("low_phase_confidence")
        if evidence.observability is None or evidence.observability_source is None:
            reasons.append("missing_observability_evidence")
        if evidence.response_reliability is None or evidence.response_source is None:
            reasons.append("missing_response_reliability")
        if evidence.contact_clear is None or evidence.contact_source is None:
            reasons.append("missing_contact_evidence")
        elif not evidence.contact_clear:
            reasons.append("contact_not_clear")

        checkpoint = None
        if not reasons:
            cfg = self.buffer.config
            if float(evidence.observability) < cfg.minimum_observability:
                reasons.append("observability_below_checkpoint_threshold")
            if float(evidence.response_reliability) < cfg.minimum_reliability:
                reasons.append("response_reliability_below_checkpoint_threshold")
            if float(joint_margin) < cfg.minimum_joint_margin:
                reasons.append("joint_margin_below_checkpoint_threshold")
            if float(singularity) < cfg.minimum_singularity_margin:
                reasons.append("singularity_margin_below_checkpoint_threshold")
        if not reasons:
            checkpoint = RecoveryCheckpoint(
                action_index=int(action_index), eef_pos=eef_pos,
                joint_pos=joint_pos, phase=str(evidence.phase),
                observability=float(evidence.observability),
                reliability=float(evidence.response_reliability),
                contact_clear=bool(evidence.contact_clear),
                joint_margin=float(joint_margin),
                singularity_margin=float(singularity),
                task_progress=float(evidence.task_progress or 0.0))
            if not self.buffer.consider(checkpoint):
                # All scalar admission gates were checked above, so a
                # remaining rejection is the temporal spacing constraint.
                reasons.append("checkpoint_spacing_gate_rejected")
                checkpoint = None

        admitted = checkpoint is not None
        row = {
            "action_index": int(action_index),
            "eef_pos": eef_pos,
            "eef_rotation": eef_rotation,
            "joint_pos": joint_pos,
            "joint_velocity": joint_velocity,
            "gripper_state": (
                float(np.mean(np.asarray(state["gripper_qpos"], dtype=float)))
                if "gripper_qpos" in state else 0.0
            ),
            "phase": evidence.phase,
            "observability": evidence.observability,
            "response_reliability": evidence.response_reliability,
            "contact_clear": evidence.contact_clear,
            "minimum_self_clearance_m": evidence.minimum_self_clearance_m,
            "minimum_environment_clearance_m": evidence.minimum_environment_clearance_m,
            "self_clearance_pair": evidence.self_clearance_pair,
            "environment_clearance_pair": evidence.environment_clearance_pair,
            "joint_margin": joint_margin,
            "singularity_margin": singularity,
            "recovery_safe": admitted,
            "checkpoint_rejection_reasons": tuple(reasons),
            "checkpoint_evidence_sources": {
                "observability": evidence.observability_source,
                "response": evidence.response_source,
                "contact": evidence.contact_source,
            },
        }
        self.history.append(row)
        return CheckpointAdmissionDecision(
            admitted, tuple(reasons), checkpoint, row)


class ConservativePrecontactPhaseTracker:
    """Expose only the high-confidence pre-contact part of a generic task.

    The tracker intentionally abstains forever after the first close command;
    it does not pretend to know whether the task is grasping, transporting or
    placing.  The admitted prefix still provides a broadly useful observation
    / approach rollback target without LIBERO task-specific labels.
    """

    def __init__(self, close_threshold: float = 0.5):
        self.close_threshold = float(close_threshold)
        self.reset()

    def reset(self):
        self.close_seen = False

    def update(self, intended_action: Sequence[float]) -> tuple[str | None, float | None]:
        action = np.asarray(intended_action, dtype=float)
        if action.size > 6 and action[6] >= self.close_threshold:
            self.close_seen = True
        if self.close_seen:
            return None, None
        return "approach", 0.8


def estimate_camera_health(images: Mapping | None) -> tuple[float | None, str | None]:
    """Low-cost sensor health, not semantic target visibility."""
    if not images:
        return None, None
    scores = []
    for key in ("agent", "wrist"):
        if key not in images:
            return None, None
        image = np.asarray(images[key])
        if image.ndim != 3 or image.shape[-1] != 3 or not np.isfinite(image).all():
            return None, None
        normalized = image.astype(float) / 255.0
        nonsaturated = float(np.mean((normalized > .01) & (normalized < .99)))
        texture = float(np.clip(np.std(normalized) / .12, 0.0, 1.0))
        scores.append(.5 * nonsaturated + .5 * texture)
    return min(scores), "paired_camera_health_v1"


def normal_response_reliability(events: Sequence[Mapping]) -> tuple[float | None, str | None]:
    """Require no alarm, but score reliability from the response expert only.

    A normal stall event may legitimately carry confidence zero because that
    confidence measures *stall evidence*, not confidence in normal robot
    response.  Mixing those semantics made every checkpoint unreliable.
    """
    informative = [
        event for event in events
        if str((event.get("evidence") or {}).get("status")) != "not_implemented"
    ]
    if not informative:
        return None, None
    if any(str(event.get("event_type")) != "normal" for event in informative):
        return None, None
    response_events = [
        event for event in informative
        if "execution_consistency" in str(event.get("source", ""))
    ]
    # Retain compatibility for callers that provide only abstract normal
    # events, while online integration always supplies named sources.
    scored = response_events or informative
    confidence = min(float(event.get("confidence", 0.0)) for event in scored)
    source = (
        "execution_consistency_explicitly_normal"
        if response_events else "all_supervisor_events_explicitly_normal"
    )
    return confidence, source


def contact_clear_from_signed_distances(self_clearance: float | None,
                                        environment_clearance: float | None, *,
                                        numerical_zero_band_m: float = 1e-5) -> bool:
    """Classify actual contact, not a generic near-obstacle risk margin.

    Positive-but-small clearance remains contact-free.  Candidate recovery
    motion is independently checked by the swept collision gate.
    """
    if numerical_zero_band_m < 0:
        raise ValueError("numerical_zero_band_m must be non-negative")
    values = (self_clearance, environment_clearance)
    return all(
        value is not None
        and np.isfinite(float(value))
        and float(value) > numerical_zero_band_m
        for value in values
    )


@dataclass(frozen=True)
class RecoveryPreparationDecision:
    ready: bool
    reason: str
    join_state: HistoricalJoinState | None = None
    rollback_target_action_index: int | None = None
    replay_references: tuple[ReverseReplayReference, ...] = ()


class OnlineRecoveryPreparer:
    """Build a recovery plan only from independently admitted checkpoints."""

    def __init__(self, selector: CheckpointSelector | None = None,
                 replay_planner: ReverseStateReplayPlanner | None = None,
                 allow_recent_unknown_reliability: bool = False,
                 dense_geometry_clearance_margin_m: float = .005):
        self.selector = selector or CheckpointSelector()
        self.replay_planner = replay_planner or ReverseStateReplayPlanner()
        self.allow_recent_unknown_reliability = bool(
            allow_recent_unknown_reliability)
        self.dense_geometry_clearance_margin_m = float(
            dense_geometry_clearance_margin_m)
        if self.dense_geometry_clearance_margin_m < 0:
            raise ValueError("dense geometry clearance margin must be non-negative")

    @staticmethod
    def _checkpoint_from_row(row: Mapping, *, reliability: float,
                             contact_clear: bool) -> RecoveryCheckpoint:
        return RecoveryCheckpoint(
            action_index=int(row["action_index"]),
            eef_pos=tuple(row["eef_pos"]),
            joint_pos=tuple(row["joint_pos"]),
            phase=str(row.get("phase") or "approach"),
            observability=float(row.get("observability") or 0.0),
            reliability=float(reliability),
            contact_clear=bool(contact_clear),
            joint_margin=float(row["joint_margin"]),
            singularity_margin=float(row["singularity_margin"]),
            task_progress=float(row.get("task_progress") or 0.0),
        )

    def _tiered_checkpoints(self, recorder: OnlineCheckpointRecorder, *,
                            failure_action_index: int) -> tuple[list[RecoveryCheckpoint],
                                                                 dict[int, str]]:
        """Recover reference points without confusing unknown evidence with danger.

        Checkpoints are navigation references, not executable commands.  Every
        rollback microstep is still checked by the instruction, swept-collision
        and pairwise post-execution gates.  Consequently the distant prefix may
        use a weaker *reference admission* contract than recent states near the
        failure, where contact and response ambiguity are more consequential.
        """
        rows = [row for row in recorder.history
                if int(row["action_index"]) < int(failure_action_index)]
        if not rows:
            return [], {}
        cfg = recorder.buffer.config
        horizon = max(1, int(failure_action_index))
        early_end = max(2, int(np.ceil(.40 * horizon)))
        middle_end = max(early_end + 1, int(np.ceil(.80 * horizon)))

        # A fixed mount/base overlap can be present at reset.  Treat only the
        # stable initial pair as structural; pair changes or worsening depth do
        # not inherit this exemption.
        prefix = rows[:min(5, len(rows))]
        pair_values: dict[tuple[str, str], list[float]] = {}
        for row in prefix:
            pair = row.get("environment_clearance_pair")
            value = row.get("minimum_environment_clearance_m")
            if pair and value is not None and np.isfinite(float(value)):
                pair_values.setdefault(tuple(pair), []).append(float(value))
        structural_pair = None
        structural_depth = None
        if pair_values:
            structural_pair, values = max(pair_values.items(), key=lambda item: len(item[1]))
            if len(values) >= max(2, len(prefix) // 2):
                structural_depth = float(np.median(values))
            else:
                structural_pair = None

        checkpoints: list[RecoveryCheckpoint] = []
        tiers: dict[int, str] = {}
        last_index = None
        for row in rows:
            index = int(row["action_index"])
            vectors_valid = all(row.get(key) is not None for key in (
                "eef_pos", "eef_rotation", "joint_pos", "joint_velocity"))
            margins_valid = (
                row.get("joint_margin") is not None
                and row.get("singularity_margin") is not None
                and float(row["joint_margin"]) >= cfg.minimum_joint_margin
                and float(row["singularity_margin"]) >= cfg.minimum_singularity_margin)
            if not vectors_valid or not margins_valid or row.get("phase") is None:
                continue

            pair = row.get("environment_clearance_pair")
            clearance = row.get("minimum_environment_clearance_m")
            structural_contact = (
                structural_pair is not None and tuple(pair or ()) == structural_pair
                and clearance is not None and structural_depth is not None
                and float(clearance) >= structural_depth - 1e-4)
            contact_ok = bool(row.get("contact_clear")) or structural_contact
            observability = row.get("observability")
            reliability = row.get("response_reliability")
            self_clearance = row.get("minimum_self_clearance_m")
            environment_clearance = row.get("minimum_environment_clearance_m")
            dense_geometry_clear = (
                self_clearance is not None
                and environment_clearance is not None
                and np.isfinite(float(self_clearance))
                and np.isfinite(float(environment_clearance))
                and float(self_clearance) >= self.dense_geometry_clearance_margin_m
                and float(environment_clearance)
                >= self.dense_geometry_clearance_margin_m)

            if dense_geometry_clear:
                # An actually observed state with calibrated geometric
                # clearance is a dense replay reference.  This certifies the
                # point, not a straight-line path to it; every connecting
                # microstep remains subject to the swept collision gate.
                tier = "geometry_clear_dense"
                admissible = contact_ok
                effective_reliability = (
                    float(reliability) if reliability is not None else .5)
            elif last_index is not None and index - last_index < cfg.minimum_spacing_steps:
                continue
            elif index < early_end:
                # Trusted startup prefix: only hard kinematic faults and a new
                # or worsening collision can reject a reference point.
                tier = "early_trusted"
                admissible = contact_ok
                effective_reliability = (
                    float(reliability) if reliability is not None else .5)
            elif index < middle_end:
                tier = "middle_permissive"
                admissible = (
                    contact_ok and observability is not None
                    and float(observability) >= .5
                    and (reliability is None or float(reliability) >= .15))
                effective_reliability = (
                    float(reliability) if reliability is not None else .4)
            else:
                tier = "recent_strict"
                admissible = (
                    contact_ok and observability is not None
                    and float(observability) >= cfg.minimum_observability
                    and (
                        (reliability is not None
                         and float(reliability) >= cfg.minimum_reliability)
                        or (reliability is None
                            and self.allow_recent_unknown_reliability)))
                effective_reliability = (
                    float(reliability) if reliability is not None else .25)
            if not admissible:
                continue
            checkpoints.append(self._checkpoint_from_row(
                row, reliability=effective_reliability, contact_clear=True))
            tiers[index] = tier
            last_index = index
        return checkpoints, tiers

    def prepare(self, recorder: OnlineCheckpointRecorder, *,
                failure_action_index: int, diagnosis: str,
                current_eef_pos: Sequence[float], current_phase: str,
                rollback_level: int = 0,
                minimum_history_depth_steps: int = 0,
                minimum_spatial_retreat_m: float = 0.0) -> RecoveryPreparationDecision:
        eligible = [
            cp for cp in recorder.buffer.checkpoints
            if cp.action_index < int(failure_action_index)
        ]
        # Strict online admissions remain preferred.  When a new task/domain
        # provides no calibrated response expert, recover a graded set of
        # historical references instead of declaring every prior state unsafe.
        tiered, tiers = self._tiered_checkpoints(
            recorder, failure_action_index=failure_action_index)
        by_index = {cp.action_index: cp for cp in eligible}
        for checkpoint in tiered:
            by_index.setdefault(checkpoint.action_index, checkpoint)
        eligible = sorted(by_index.values(), key=lambda cp: cp.action_index)
        if len(eligible) < 2:
            return RecoveryPreparationDecision(False, "insufficient_admitted_checkpoints")
        join_checkpoint = max(eligible, key=lambda cp: cp.action_index)
        ranked = self.selector.ranked(
            eligible, diagnosis=diagnosis, current_eef_pos=current_eef_pos,
            current_phase=current_phase,
            before_action_index=join_checkpoint.action_index)
        minimum_history_depth_steps = max(0, int(minimum_history_depth_steps))
        minimum_spatial_retreat_m = max(0.0, float(minimum_spatial_retreat_m))
        failure_eef = np.asarray(current_eef_pos, dtype=float)
        ranked = [checkpoint for checkpoint in ranked
                  if (join_checkpoint.action_index - checkpoint.action_index
                      >= minimum_history_depth_steps
                      and np.linalg.norm(
                          np.asarray(checkpoint.eef_pos, dtype=float) - failure_eef)
                      >= minimum_spatial_retreat_m)]
        target = (ranked[rollback_level]
                  if rollback_level < len(ranked) else None)
        if target is None:
            return RecoveryPreparationDecision(False, "no_earlier_compatible_checkpoint")
        eligible_indexes = {cp.action_index for cp in eligible}
        rows = {int(row["action_index"]): row for row in recorder.history
                if int(row["action_index"]) in eligible_indexes}
        join_row = rows.get(join_checkpoint.action_index)
        if join_row is None or not join_row.get("eef_rotation"):
            return RecoveryPreparationDecision(False, "join_checkpoint_missing_pose")
        replay_history = [
            ({**row, "recovery_safe": True,
              "reference_admission_tier": tiers.get(int(row["action_index"]), "strict")}
             if int(row["action_index"]) in eligible_indexes else row)
            for row in recorder.history
        ]
        references = self.replay_planner.build(
            replay_history, join_action_index=join_checkpoint.action_index,
            target_action_index=target.action_index)
        if len(references) < 2:
            return RecoveryPreparationDecision(False, "insufficient_reverse_replay_references")
        join_state = HistoricalJoinState(
            action_index=join_checkpoint.action_index,
            eef_pos=tuple(join_row["eef_pos"]),
            joint_pos=tuple(join_row["joint_pos"]),
            eef_rotation=tuple(join_row["eef_rotation"]),
            joint_velocity=tuple(join_row["joint_velocity"]),
            gripper_state=float(join_row.get("gripper_state", 0.0)))
        return RecoveryPreparationDecision(
            True, "ready", join_state, target.action_index, references)
