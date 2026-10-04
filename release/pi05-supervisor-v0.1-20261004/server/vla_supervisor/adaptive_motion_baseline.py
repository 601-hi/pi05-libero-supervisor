"""Leakage-resistant relative visual-motion evidence.

This module produces standardized measurements, not anomaly probabilities.
Only explicitly trusted observations may update the rolling reference.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RelativeMotionEvidence:
    ready: bool
    samples: int
    current_log_motion: float | None
    reference_median: float | None
    robust_scale: float | None
    motion_drop_z: float | None
    motion_excess_z: float | None
    update_accepted: bool
    status: str


class AdaptiveMotionBaseline:
    """Compare motion with a trusted rolling baseline using median and MAD."""

    def __init__(self, *, window: int = 50, minimum_samples: int = 12,
                 minimum_scale: float = 0.02, minimum_background_confidence: float = 0.4):
        if window < 2 or not 2 <= minimum_samples <= window:
            raise ValueError("minimum_samples must be between 2 and window")
        if minimum_scale <= 0 or not 0 <= minimum_background_confidence <= 1:
            raise ValueError("scale must be positive and confidence must be in [0, 1]")
        self.values = deque(maxlen=int(window))
        self.minimum_samples = int(minimum_samples)
        self.minimum_scale = float(minimum_scale)
        self.minimum_background_confidence = float(minimum_background_confidence)

    def reset(self) -> None:
        self.values.clear()

    def _reference(self) -> tuple[float, float] | None:
        if len(self.values) < self.minimum_samples:
            return None
        values = np.asarray(self.values, dtype=float)
        median = float(np.median(values))
        mad_scale = float(1.4826 * np.median(np.abs(values - median)))
        return median, max(mad_scale, self.minimum_scale)

    def observe(self, motion_px: float | None, *, background_confidence: float,
                allow_reference_update: bool) -> RelativeMotionEvidence:
        if motion_px is None or not np.isfinite(motion_px) or motion_px < 0:
            return RelativeMotionEvidence(
                False, len(self.values), None, None, None, None, None, False, "invalid_measurement"
            )
        current = float(np.log1p(motion_px))
        reference = self._reference()
        drop = excess = median = scale = None
        if reference is not None:
            median, scale = reference
            drop = (median - current) / scale
            excess = (current - median) / scale

        accepted = bool(
            allow_reference_update
            and np.isfinite(background_confidence)
            and background_confidence >= self.minimum_background_confidence
        )
        if accepted:
            self.values.append(current)
        ready = reference is not None
        status = "scored" if ready else ("warming_up" if accepted else "reference_update_refused")
        return RelativeMotionEvidence(
            ready, len(self.values), current, median, scale, drop, excess, accepted, status
        )

