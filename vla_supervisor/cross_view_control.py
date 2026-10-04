"""Cross-check world motion in a fixed view with attachment in a wrist view."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class CrossViewState(str, Enum):
    CONFIRMED_ATTACHED = "confirmed_attached"
    STATIC_IN_WORLD = "static_in_world"
    FIXED_ONLY_MOTION = "fixed_only_motion"
    WRIST_ONLY_MOTION = "wrist_only_motion"
    CONFLICT = "conflict"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class CrossViewEvidence:
    fixed_world_motion: bool | None
    wrist_attachment_stable: bool | None
    fixed_confidence: float
    wrist_confidence: float


@dataclass(frozen=True)
class CrossViewDecision:
    state: CrossViewState
    confidence: float
    request_more_evidence: bool


def cross_validate_views(
    evidence: CrossViewEvidence,
    *,
    minimum_view_confidence: float = 0.5,
) -> CrossViewDecision:
    """Fuse complementary facts without forcing agreement from invalid views."""
    for value in (evidence.fixed_confidence, evidence.wrist_confidence, minimum_view_confidence):
        if not 0 <= value <= 1:
            raise ValueError("confidences must lie in [0, 1]")
    fixed_valid = evidence.fixed_confidence >= minimum_view_confidence and evidence.fixed_world_motion is not None
    wrist_valid = evidence.wrist_confidence >= minimum_view_confidence and evidence.wrist_attachment_stable is not None
    if not fixed_valid and not wrist_valid:
        return CrossViewDecision(CrossViewState.UNKNOWN, 0.0, True)
    if fixed_valid and wrist_valid:
        confidence = min(evidence.fixed_confidence, evidence.wrist_confidence)
        if evidence.fixed_world_motion and evidence.wrist_attachment_stable:
            return CrossViewDecision(CrossViewState.CONFIRMED_ATTACHED, confidence, False)
        if not evidence.fixed_world_motion and not evidence.wrist_attachment_stable:
            return CrossViewDecision(CrossViewState.STATIC_IN_WORLD, confidence, False)
        return CrossViewDecision(CrossViewState.CONFLICT, confidence, True)
    if fixed_valid:
        return CrossViewDecision(
            CrossViewState.FIXED_ONLY_MOTION if evidence.fixed_world_motion else CrossViewState.STATIC_IN_WORLD,
            evidence.fixed_confidence,
            True,
        )
    return CrossViewDecision(
        CrossViewState.WRIST_ONLY_MOTION if evidence.wrist_attachment_stable else CrossViewState.UNKNOWN,
        evidence.wrist_confidence,
        True,
    )

