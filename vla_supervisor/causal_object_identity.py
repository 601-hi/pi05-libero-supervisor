"""Causal locking of the visually manipulated object among tracked candidates."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ObjectIdentityState:
    locked_object_id: int | None
    leading_object_id: int | None
    leading_motion: float
    motion_margin: float
    confirmation_streak: int


class CausalMotionIdentitySelector:
    """Lock a candidate only after sustained, separated image-plane motion."""

    def __init__(
        self,
        image_shape: tuple[int, int],
        minimum_motion: float = 0.03,
        minimum_margin: float = 0.02,
        confirmation_frames: int = 3,
    ) -> None:
        if len(image_shape) != 2 or any(size <= 0 for size in image_shape):
            raise ValueError("image_shape must contain two positive dimensions")
        if minimum_motion < 0 or minimum_margin < 0:
            raise ValueError("motion thresholds must be non-negative")
        if confirmation_frames < 1:
            raise ValueError("confirmation_frames must be at least one")
        self._diagonal = float(np.hypot(*image_shape))
        self.minimum_motion = minimum_motion
        self.minimum_margin = minimum_margin
        self.confirmation_frames = confirmation_frames
        self._origins: dict[int, np.ndarray] = {}
        self._maximum_motion: dict[int, float] = {}
        self._streak_id: int | None = None
        self._streak = 0
        self._locked_id: int | None = None

    def update(self, centroids_xy: dict[int, tuple[float, float] | None]) -> ObjectIdentityState:
        for object_id, value in centroids_xy.items():
            if value is None:
                continue
            point = np.asarray(value, dtype=float)
            if point.shape != (2,) or not np.all(np.isfinite(point)):
                continue
            self._origins.setdefault(object_id, point.copy())
            displacement = float(np.linalg.norm(point - self._origins[object_id]) / self._diagonal)
            self._maximum_motion[object_id] = max(self._maximum_motion.get(object_id, 0.0), displacement)

        ranked = sorted(
            ((motion, object_id) for object_id, motion in self._maximum_motion.items()), reverse=True
        )
        if not ranked:
            return ObjectIdentityState(self._locked_id, None, 0.0, 0.0, self._streak)
        leading_motion, leading_id = ranked[0]
        runner_up_motion = ranked[1][0] if len(ranked) > 1 else leading_motion
        margin = leading_motion - runner_up_motion

        qualifies = (
            len(ranked) > 1
            and leading_motion >= self.minimum_motion
            and margin >= self.minimum_margin
        )
        if self._locked_id is None:
            if qualifies and leading_id == self._streak_id:
                self._streak += 1
            elif qualifies:
                self._streak_id, self._streak = leading_id, 1
            else:
                self._streak_id, self._streak = None, 0
            if self._streak >= self.confirmation_frames:
                self._locked_id = leading_id
        return ObjectIdentityState(
            self._locked_id, leading_id, leading_motion, margin, self._streak
        )
