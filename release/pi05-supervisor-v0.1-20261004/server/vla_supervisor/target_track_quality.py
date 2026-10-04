"""Robot- and task-agnostic quality gate for a propagated target mask."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TargetTrackQuality:
    usable: bool
    score: float
    area_ratio: float
    centroid_jump_fraction: float
    reason: str


def _mask_properties(mask):
    value = np.asarray(mask, dtype=bool)
    if value.ndim != 2:
        raise ValueError("target mask must be 2D")
    points = np.argwhere(value)
    if not len(points):
        return value, 0, None
    return value, int(len(points)), points.mean(axis=0)


def assess_target_track(reference_mask, previous_mask, current_mask, *,
                        minimum_area_ratio: float = .20,
                        maximum_area_ratio: float = 5.0,
                        maximum_centroid_jump_fraction: float = .18):
    """Reject disappearance, segmentation collapse and implausible frame jumps.

    The score is a bounded engineering quality measure, not a probability of
    semantic correctness.  Displacement is normalized by the image diagonal,
    making the thresholds portable across camera resolutions.
    """
    if not 0 < minimum_area_ratio <= 1 or maximum_area_ratio < 1:
        raise ValueError("invalid area-ratio bounds")
    if not 0 < maximum_centroid_jump_fraction <= 1:
        raise ValueError("invalid centroid jump bound")
    reference, reference_area, _ = _mask_properties(reference_mask)
    previous, previous_area, previous_center = _mask_properties(previous_mask)
    current, current_area, current_center = _mask_properties(current_mask)
    if reference.shape != previous.shape or reference.shape != current.shape:
        raise ValueError("track masks must share shape")
    if not reference_area or not previous_area or not current_area:
        return TargetTrackQuality(False, 0.0, 0.0, float("inf"), "empty_mask")
    area_ratio = current_area / reference_area
    diagonal = float(np.hypot(*reference.shape))
    jump = float(np.linalg.norm(current_center - previous_center) / diagonal)
    area_low_quality = min(1.0, area_ratio / minimum_area_ratio)
    area_high_quality = min(1.0, maximum_area_ratio / area_ratio)
    jump_quality = max(0.0, 1.0 - jump / maximum_centroid_jump_fraction)
    score = float(min(area_low_quality, area_high_quality, jump_quality))
    if area_ratio < minimum_area_ratio:
        reason = "mask_area_collapse"
    elif area_ratio > maximum_area_ratio:
        reason = "mask_area_explosion"
    elif jump > maximum_centroid_jump_fraction:
        reason = "implausible_centroid_jump"
    else:
        reason = "usable"
    return TargetTrackQuality(reason == "usable", score, float(area_ratio), jump, reason)
