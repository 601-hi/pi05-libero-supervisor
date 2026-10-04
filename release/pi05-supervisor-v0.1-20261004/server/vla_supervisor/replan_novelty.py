"""Action-trajectory novelty gate for post-rollback policy replans."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class ReplanNoveltyDecision:
    state: str
    accept: bool
    aligned_steps: int
    relative_l2: float | None
    median_direction_cosine: float | None
    integrated_direction_cosine: float | None
    gripper_sign_agreement: float | None
    reason: str


class ReplanNoveltyGate:
    """Reject a new chunk only when several geometric tests agree it repeats.

    This gate does not judge whether an action is semantically correct.  It
    answers the narrower causal question: did replanning materially change the
    local command trajectory that just led to recovery?
    """

    def __init__(self, *, minimum_steps: int = 3,
                 maximum_relative_l2: float = .20,
                 minimum_median_cosine: float = .97,
                 minimum_integrated_cosine: float = .95,
                 minimum_gripper_agreement: float = .80):
        self.minimum_steps = int(minimum_steps)
        self.maximum_relative_l2 = float(maximum_relative_l2)
        self.minimum_median_cosine = float(minimum_median_cosine)
        self.minimum_integrated_cosine = float(minimum_integrated_cosine)
        self.minimum_gripper_agreement = float(minimum_gripper_agreement)

    def compare(self, failed_chunk: Sequence[Sequence[float]] | None,
                new_chunk: Sequence[Sequence[float]] | None) -> ReplanNoveltyDecision:
        if failed_chunk is None or new_chunk is None:
            return ReplanNoveltyDecision(
                "insufficient_evidence", True, 0, None, None, None, None,
                "missing_reference_chunk")
        old = np.asarray(failed_chunk, dtype=float)
        new = np.asarray(new_chunk, dtype=float)
        if old.ndim != 2 or new.ndim != 2 or old.shape[1] < 7 or new.shape[1] < 7:
            return ReplanNoveltyDecision(
                "insufficient_evidence", True, 0, None, None, None, None,
                "invalid_chunk_shape")
        count = min(len(old), len(new))
        if count < self.minimum_steps:
            return ReplanNoveltyDecision(
                "insufficient_evidence", True, count, None, None, None, None,
                "too_few_aligned_steps")

        old_motion = old[:count, :6]
        new_motion = new[:count, :6]
        relative_l2 = float(
            np.linalg.norm(new_motion - old_motion)
            / (np.linalg.norm(old_motion) + 1e-12))
        denominators = (
            np.linalg.norm(old_motion, axis=1)
            * np.linalg.norm(new_motion, axis=1) + 1e-12)
        cosines = np.sum(old_motion * new_motion, axis=1) / denominators
        median_cosine = float(np.median(cosines))
        old_integrated = old_motion.sum(axis=0)
        new_integrated = new_motion.sum(axis=0)
        integrated_cosine = float(
            np.dot(old_integrated, new_integrated)
            / (np.linalg.norm(old_integrated) * np.linalg.norm(new_integrated) + 1e-12))
        gripper_agreement = float(np.mean(
            np.sign(old[:count, 6]) == np.sign(new[:count, 6])))

        repeated = (
            relative_l2 <= self.maximum_relative_l2
            and median_cosine >= self.minimum_median_cosine
            and integrated_cosine >= self.minimum_integrated_cosine
            and gripper_agreement >= self.minimum_gripper_agreement
        )
        return ReplanNoveltyDecision(
            "repeated" if repeated else "changed",
            not repeated,
            count,
            relative_l2,
            median_cosine,
            integrated_cosine,
            gripper_agreement,
            "replan_matches_failed_chunk" if repeated else "material_trajectory_change",
        )

