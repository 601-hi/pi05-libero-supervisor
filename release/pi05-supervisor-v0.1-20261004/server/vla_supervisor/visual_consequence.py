"""Online fixed/wrist visual consequence evidence with explicit abstention.

The scorer deliberately separates measurement from calibration.  Optical-flow
residuals are not probabilities; actionable fields remain ``None`` unless a
candidate provider and frozen calibration callbacks are supplied.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Any

import numpy as np

from .adaptive_motion_baseline import AdaptiveMotionBaseline
from .background_motion import estimate_background_motion, residual_motion_in_mask


CandidateProvider = Callable[..., Mapping[str, Any]]
ProbabilityCalibrator = Callable[[Mapping[str, float]], float]


@dataclass(frozen=True)
class VisualConsequenceConfig:
    gripper_closed_threshold: float = 0.5
    manipulation_window_steps: int = 40

    def __post_init__(self):
        if not 0 <= self.gripper_closed_threshold <= 1:
            raise ValueError("gripper threshold must be in [0, 1]")
        if self.manipulation_window_steps < 1:
            raise ValueError("manipulation window must be positive")


class BackgroundConsequenceEvidenceScorer:
    """Convert paired RGB frames into calibrated contact-consequence evidence.

    ``candidate_provider`` may return ``candidate_mask``,
    ``robot_exclusion_mask``, ``gripper_candidate_proximity``,
    ``controlled_object_candidate_probability``, ``attachment_probability``
    and ``semantic_target_probability``. Candidate quality is deliberately
    separate from physical attachment. Raw
    residual motion is exposed through ``last_diagnostics`` only.  It becomes
    an evidence probability solely through an explicitly supplied frozen
    calibrator.
    """

    def __init__(
        self,
        candidate_provider: CandidateProvider | None = None,
        *,
        motion_calibrator: ProbabilityCalibrator | None = None,
        blocked_calibrator: ProbabilityCalibrator | None = None,
        adaptive_motion_baseline: AdaptiveMotionBaseline | None = None,
        config: VisualConsequenceConfig = VisualConsequenceConfig(),
    ):
        self.candidate_provider = candidate_provider
        self.motion_calibrator = motion_calibrator
        self.blocked_calibrator = blocked_calibrator
        self.adaptive_motion_baseline = adaptive_motion_baseline
        self.config = config
        self.reset()

    def reset(self):
        self.previous_gripper_sign = None
        self.manipulation_remaining = 0
        self.last_diagnostics = {"status": "reset"}
        reset = getattr(self.candidate_provider, "reset", None)
        if reset is not None:
            reset()
        if self.adaptive_motion_baseline is not None:
            self.adaptive_motion_baseline.reset()

    def _phase(self, intended_action) -> tuple[bool, bool]:
        command = float(intended_action[6]) if len(intended_action) > 6 else 0.0
        threshold = self.config.gripper_closed_threshold
        sign = 1 if command >= threshold else -1 if command <= -threshold else 0
        transition = (
            self.previous_gripper_sign in {-1, 1}
            and sign in {-1, 1}
            and sign != self.previous_gripper_sign
        )
        if sign in {-1, 1}:
            self.previous_gripper_sign = sign
        if transition:
            self.manipulation_remaining = self.config.manipulation_window_steps
        active = self.manipulation_remaining > 0
        if active:
            self.manipulation_remaining -= 1
        return active, sign > 0

    @staticmethod
    def _clip_optional(value):
        if value is None:
            return None
        return float(np.clip(float(value), 0.0, 1.0))

    def __call__(self, images_before, images_after, intended_action, history):
        manipulation_phase, gripper_closed = self._phase(intended_action)
        base = {
            "manipulation_phase": manipulation_phase,
            "gripper_closed": gripper_closed,
            "background_confidence": 0.0,
            "controlled_object_candidate_probability": None,
            "gripper_candidate_proximity": None,
            "independent_motion_probability": None,
            "attachment_probability": None,
            "blocked_motion_probability": None,
            "semantic_target_probability": None,
        }
        if not images_before or not images_after:
            self.last_diagnostics = {"status": "missing_paired_images"}
            return base
        fixed_before = images_before.get("agent")
        fixed_after = images_after.get("agent")
        if fixed_before is None or fixed_after is None:
            self.last_diagnostics = {"status": "missing_fixed_view"}
            return base

        candidate = {}
        if self.candidate_provider is not None:
            candidate = dict(self.candidate_provider(
                images_before, images_after, intended_action, history
            ))
        exclusion = candidate.get("robot_exclusion_mask")
        estimate = estimate_background_motion(fixed_before, fixed_after, exclusion)
        base["background_confidence"] = estimate.confidence
        self.last_diagnostics = {
            "status": "background_only",
            "background_valid": estimate.valid,
            "tracked_points": estimate.tracked_points,
            "inlier_ratio": estimate.inlier_ratio,
            "median_reprojection_error_px": estimate.median_reprojection_error_px,
            "spatial_coverage": estimate.spatial_coverage,
            "background_confidence": estimate.confidence,
        }
        mask = candidate.get("candidate_mask")
        if mask is None or not estimate.valid:
            self.last_diagnostics["abstention_reason"] = (
                "candidate_mask_unavailable" if mask is None else "background_invalid"
            )
            return base

        residual = residual_motion_in_mask(fixed_before, fixed_after, mask, estimate)
        features = {
            "median_residual_px": residual.median_px,
            "p90_residual_px": residual.p90_px,
            "coherent_fraction": residual.coherent_fraction,
            "valid_pixels": float(residual.valid_pixels),
            "translation_command_norm": float(np.linalg.norm(np.asarray(intended_action[:3], float))),
            "background_confidence": estimate.confidence,
        }
        self.last_diagnostics.update(features)
        self.last_diagnostics["status"] = "candidate_measured"
        if self.adaptive_motion_baseline is not None:
            relative = self.adaptive_motion_baseline.observe(
                residual.median_px,
                background_confidence=estimate.confidence,
                allow_reference_update=bool(candidate.get("allow_reference_update", False)),
            )
            self.last_diagnostics["relative_motion"] = {
                "ready": relative.ready,
                "samples": relative.samples,
                "motion_drop_z": relative.motion_drop_z,
                "motion_excess_z": relative.motion_excess_z,
                "update_accepted": relative.update_accepted,
                "status": relative.status,
            }
        if self.motion_calibrator is not None:
            base["independent_motion_probability"] = self._clip_optional(
                self.motion_calibrator(features)
            )
        if self.blocked_calibrator is not None:
            base["blocked_motion_probability"] = self._clip_optional(
                self.blocked_calibrator(features)
            )
        for key in (
            "controlled_object_candidate_probability",
            "gripper_candidate_proximity", "attachment_probability",
            "semantic_target_probability",
        ):
            base[key] = self._clip_optional(candidate.get(key))
        return base
