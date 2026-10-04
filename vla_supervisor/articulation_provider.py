"""Fail-closed adapters for online articulated-relation measurements.

The adapter intentionally knows nothing about a particular detector or
tracker.  A perception backend returns a small mapping; this module validates
that mapping and converts it into the typed measurement consumed by the
temporal articulation watchdog.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np

from .articulation_progress import ArticulationMeasurement


@dataclass(frozen=True)
class ArticulationRelationContext:
    relation_id: str
    predicate: str
    calibration_id: str
    provenance: str

    def __post_init__(self) -> None:
        if not all((self.relation_id, self.predicate, self.calibration_id,
                    self.provenance)):
            raise ValueError("articulation relation context fields are required")


class ValidatedArticulationProvider:
    """Adapt a detector/tracker callback without granting it control authority.

    The callback may return ``None`` or a mapping with ``area_fraction``,
    ``confidence``, ``observable``, ``identity_reliable`` and ``calibrated``.
    Missing, malformed or non-finite values become an explicit unobservable
    measurement; they never become a positive or negative task conclusion.
    """

    def __init__(self, backend: Callable, context: ArticulationRelationContext):
        self.backend = backend
        self.context = context
        self._last_action_index: int | None = None

    def reset(self) -> None:
        self._last_action_index = None
        reset = getattr(self.backend, "reset", None)
        if reset is not None:
            reset()

    def __call__(self, images_before, images_after, intended_action, history,
                 action_index) -> ArticulationMeasurement:
        stale = (self._last_action_index is not None
                 and action_index <= self._last_action_index)
        self._last_action_index = action_index
        try:
            raw = self.backend(
                images_before=images_before,
                images_after=images_after,
                intended_action=intended_action,
                history=history,
                action_index=action_index,
            )
        except (KeyError, TypeError, ValueError, FloatingPointError):
            raw = None
        return self._coerce(raw, action_index=action_index, stale=stale)

    def _coerce(self, raw, *, action_index: int,
                stale: bool) -> ArticulationMeasurement:
        valid_mapping = isinstance(raw, Mapping)
        area = raw.get("area_fraction") if valid_mapping else None
        confidence = raw.get("confidence", 0.0) if valid_mapping else 0.0
        try:
            area = None if area is None else float(area)
            confidence = float(confidence)
        except (TypeError, ValueError):
            area, confidence = None, 0.0
        finite_area = area is not None and np.isfinite(area) and area > 0.0
        finite_confidence = np.isfinite(confidence)
        confidence = float(np.clip(confidence, 0.0, 1.0)) if finite_confidence else 0.0
        observable = bool(valid_mapping and raw.get("observable", False)
                          and finite_area and not stale)
        identity_reliable = bool(valid_mapping and raw.get("identity_reliable", False)
                                 and not stale)
        calibrated = bool(valid_mapping and raw.get("calibrated", False)
                          and raw.get("calibration_id", self.context.calibration_id)
                          == self.context.calibration_id and not stale)
        provenance = (str(raw.get("provenance")) if valid_mapping
                      and raw.get("provenance") else self.context.provenance)
        return ArticulationMeasurement(
            action_index=int(action_index),
            area_fraction=area if finite_area else None,
            confidence=confidence,
            observable=observable,
            identity_reliable=identity_reliable,
            calibrated=calibrated,
            calibration_id=self.context.calibration_id,
            relation_id=self.context.relation_id,
            predicate=self.context.predicate,
            provenance=provenance,
        )
