"""Conservative fusion of multi-close physical evidence."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ControlEvidenceState(str, Enum):
    INSUFFICIENT_WINDOW = "insufficient_window"
    AMBIGUOUS = "ambiguous"
    WRIST_CANDIDATE_ONLY = "wrist_candidate_only"
    FIXED_MOTION_ONLY = "fixed_motion_only"
    PREEXISTING_OR_ROBOT_MOTION = "preexisting_or_robot_motion"
    POSSIBLE_NEW_CONTROL = "possible_new_control"
    EVIDENCE_SPLIT_ACROSS_CANDIDATES = "evidence_split_across_candidates"


@dataclass(frozen=True)
class ControlEvidence:
    state: ControlEvidenceState
    establish_control: bool
    next_evidence_request: str


def classify_control_evidence(
    *, wrist_state: str, wrist_passes_gate: bool,
    fixed_has_sustained_motion: bool, best_sustained_change: float | None,
    same_candidate_semantic_alignment_count: int | None = None,
) -> ControlEvidence:
    """Fuse independent cues without equating cross-view co-occurrence to identity.

    A positive change is only a descriptive sign test (relative motion drops
    while global motion rises).  It produces POSSIBLE evidence, never an
    established-control state, until cross-view identity is confirmed.
    """
    if wrist_state == "insufficient_post_window":
        return ControlEvidence(ControlEvidenceState.INSUFFICIENT_WINDOW, False, "collect_longer_postclose_window")
    if wrist_passes_gate and fixed_has_sustained_motion:
        if best_sustained_change is not None and best_sustained_change > 0:
            if same_candidate_semantic_alignment_count == 0:
                return ControlEvidence(
                    ControlEvidenceState.EVIDENCE_SPLIT_ACROSS_CANDIDATES,
                    False,
                    "do_not_merge_motion_and_semantics_from_different_candidates",
                )
            return ControlEvidence(ControlEvidenceState.POSSIBLE_NEW_CONTROL, False,
                                   "confirm_cross_view_identity_semantics_and_persistence")
        return ControlEvidence(ControlEvidenceState.PREEXISTING_OR_ROBOT_MOTION, False,
                               "reject_robot_pixels_or_seek_independent_object_motion")
    if wrist_passes_gate:
        return ControlEvidence(ControlEvidenceState.WRIST_CANDIDATE_ONLY, False,
                               "seek_fixed_view_world_motion")
    if fixed_has_sustained_motion:
        return ControlEvidence(ControlEvidenceState.FIXED_MOTION_ONLY, False,
                               "resolve_wrist_candidate_and_cross_view_identity")
    return ControlEvidence(ControlEvidenceState.AMBIGUOUS, False, "request_more_evidence")
