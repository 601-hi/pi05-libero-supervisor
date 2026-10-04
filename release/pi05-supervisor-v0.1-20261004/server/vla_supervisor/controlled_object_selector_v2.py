"""Online controlled-object identity from gripper, background and mask evidence."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Mapping

import numpy as np


@dataclass(frozen=True)
class CandidateControlEvidence:
    centroid_xy: tuple[float, float] | None
    background_residual_normalized: float | None
    robot_overlap_fraction: float = 0.0
    preclose_robot_motion_likeness: float = 0.0
    visible: bool = True


@dataclass(frozen=True)
class ControlledObjectDecision:
    controlled_object_id: Hashable | None
    leading_object_id: Hashable | None
    state: str
    score_margin: float
    confirmation_streak: int
    evidence_valid: bool


class ControlledObjectSelectorV2:
    """Require post-close, near-gripper motion independent of the background.

    Gripper pixels may come from an RGB detector or from an independently
    calibrated EEF-to-pixel projection.  Simulator camera matrices and object
    state are intentionally absent from this interface.
    """

    def __init__(
        self,
        *,
        image_shape: tuple[int, int] = (224, 224),
        close_threshold: float = 0.5,
        open_threshold: float = -0.5,
        proximity_radius_px: float = 30.0,
        minimum_displacement_normalized: float = 0.01,
        minimum_background_residual_normalized: float = 0.001,
        maximum_robot_overlap_fraction: float = 0.6,
        maximum_preclose_robot_motion_likeness: float = 0.85,
        minimum_score_margin: float = 0.005,
        confirmation_frames: int = 3,
        grace_motion_frames: int = 20,
    ) -> None:
        if open_threshold >= close_threshold:
            raise ValueError("open_threshold must be less than close_threshold")
        if len(image_shape) != 2 or any(value <= 0 for value in image_shape):
            raise ValueError("image_shape must contain two positive dimensions")
        if proximity_radius_px <= 0 or confirmation_frames < 1 or grace_motion_frames < 1:
            raise ValueError("radii and persistence parameters must be positive")
        self.close_threshold = close_threshold
        self.image_diagonal = float(np.hypot(*image_shape))
        self.open_threshold = open_threshold
        self.proximity_radius_px = proximity_radius_px
        self.minimum_displacement = minimum_displacement_normalized
        self.minimum_background_residual = minimum_background_residual_normalized
        self.maximum_robot_overlap = maximum_robot_overlap_fraction
        self.maximum_preclose_robot_motion_likeness = maximum_preclose_robot_motion_likeness
        self.minimum_margin = minimum_score_margin
        self.confirmation_required = confirmation_frames
        self.grace_motion_frames = grace_motion_frames
        self.reset()

    def reset(self) -> None:
        self._closing = False
        self._origins: dict[Hashable, np.ndarray] = {}
        self._previous_candidates: dict[Hashable, np.ndarray] = {}
        self._previous_gripper: np.ndarray | None = None
        self._motion_frames = 0
        self._streak_id: Hashable | None = None
        self._streak = 0
        self._locked_id: Hashable | None = None

    def _decision(self, leading, state, margin=0.0, valid=True) -> ControlledObjectDecision:
        return ControlledObjectDecision(
            self._locked_id, leading, state, float(margin), self._streak, valid
        )

    def update(
        self,
        *,
        gripper_command: float,
        gripper_moving: bool,
        gripper_xy: tuple[float, float] | None,
        projection_uncertainty_px: float,
        candidates: Mapping[Hashable, CandidateControlEvidence],
        background_confidence: float,
    ) -> ControlledObjectDecision:
        if gripper_command <= self.open_threshold:
            self.reset()
            return self._decision(None, "open")
        if gripper_command >= self.close_threshold and not self._closing:
            self._closing = True
            self._origins.clear()
            self._motion_frames = 0
            self._streak_id = None
            self._streak = 0
            self._locked_id = None
        if not self._closing:
            return self._decision(None, "open")
        if gripper_xy is None or background_confidence <= 0:
            return self._decision(None, "unknown", valid=False)

        gripper = np.asarray(gripper_xy, dtype=float)
        if gripper.shape != (2,) or not np.all(np.isfinite(gripper)):
            return self._decision(None, "unknown", valid=False)
        radius = self.proximity_radius_px + max(float(projection_uncertainty_px), 0.0)
        scores = []
        current_points: dict[Hashable, np.ndarray] = {}
        for candidate_id, evidence in candidates.items():
            if not evidence.visible or evidence.centroid_xy is None:
                continue
            point = np.asarray(evidence.centroid_xy, dtype=float)
            if point.shape != (2,) or not np.all(np.isfinite(point)):
                continue
            current_points[candidate_id] = point
            self._origins.setdefault(candidate_id, point.copy())
            if evidence.robot_overlap_fraction > self.maximum_robot_overlap:
                continue
            if evidence.preclose_robot_motion_likeness > self.maximum_preclose_robot_motion_likeness:
                continue
            distance = float(np.linalg.norm(point - gripper))
            if distance > radius:
                continue
            displacement = float(np.linalg.norm(point - self._origins[candidate_id]))
            residual = evidence.background_residual_normalized
            if residual is None or not np.isfinite(residual):
                continue
            # Image diagonal normalization is supplied by the residual source;
            # displacement is converted using the conservative 224px default
            # scale only for the online ranking threshold.
            displacement_normalized = displacement / self.image_diagonal
            if displacement_normalized < self.minimum_displacement or residual < self.minimum_background_residual:
                continue
            synchrony = 0.0
            previous = self._previous_candidates.get(candidate_id)
            if previous is not None and self._previous_gripper is not None:
                object_delta = point - previous
                gripper_delta = gripper - self._previous_gripper
                denominator = np.linalg.norm(object_delta) * np.linalg.norm(gripper_delta)
                if denominator > 1e-6:
                    synchrony = float(np.dot(object_delta, gripper_delta) / denominator)
            score = displacement_normalized + 0.25 * residual + 0.01 * max(synchrony, 0.0)
            scores.append((score, candidate_id))

        self._previous_candidates = current_points
        self._previous_gripper = gripper
        if gripper_moving:
            self._motion_frames += 1
        scores.sort(reverse=True, key=lambda item: item[0])
        if not scores:
            state = "grasp_not_established" if self._motion_frames >= self.grace_motion_frames else "waiting_for_effect"
            return self._decision(None, state)
        leading_score, leading_id = scores[0]
        runner_up = scores[1][0] if len(scores) > 1 else 0.0
        margin = leading_score - runner_up
        qualifies = margin >= self.minimum_margin
        if self._locked_id is None:
            if qualifies and leading_id == self._streak_id:
                self._streak += 1
            elif qualifies:
                self._streak_id, self._streak = leading_id, 1
            else:
                self._streak_id, self._streak = None, 0
            if self._streak >= self.confirmation_required:
                self._locked_id = leading_id
        state = "controlled" if self._locked_id is not None else ("candidate" if qualifies else "ambiguous")
        return self._decision(leading_id, state, margin)
