"""Interpret visual grasp-control states with independent execution evidence."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .events import RecommendedAction
from .grasp_control_state import GraspControlResult, GraspControlState


class GraspDiagnosis(str, Enum):
    NORMAL_OR_PENDING = "normal_or_pending"
    VISUAL_UNKNOWN = "visual_unknown"
    POSSIBLE_PUSH_OR_COLLISION = "possible_push_or_collision"
    MISSED_OR_EMPTY_GRASP = "missed_or_empty_grasp"
    POSSIBLE_OBSTRUCTION = "possible_obstruction"
    POSSIBLE_DROP = "possible_drop"


@dataclass(frozen=True)
class GraspSafetyDecision:
    diagnosis: GraspDiagnosis
    action: RecommendedAction
    confidence: float
    evidence: dict


def fuse_grasp_safety(
    grasp: GraspControlResult,
    *,
    visual_confidence: float,
    execution_mismatch: bool,
    execution_confidence: float,
) -> GraspSafetyDecision:
    for name, value in (("visual_confidence", visual_confidence),
                        ("execution_confidence", execution_confidence)):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]")
    evidence = {
        "grasp_control_state": grasp.state.value,
        "controlled_object_id": grasp.controlled_object_id,
        "informative_motion_frames": grasp.informative_motion_frames,
        "visual_evidence_valid": grasp.evidence_valid,
        "execution_mismatch": bool(execution_mismatch),
    }
    if not grasp.evidence_valid or grasp.state is GraspControlState.UNKNOWN:
        return GraspSafetyDecision(
            GraspDiagnosis.VISUAL_UNKNOWN, RecommendedAction.REQUEST_MORE_EVIDENCE,
            visual_confidence, evidence,
        )
    if grasp.state is GraspControlState.CONTROL_LOST:
        return GraspSafetyDecision(
            GraspDiagnosis.POSSIBLE_DROP, RecommendedAction.STOP_AND_REPLAN,
            visual_confidence, evidence,
        )
    if grasp.state is GraspControlState.GRASP_NOT_ESTABLISHED:
        if execution_mismatch and execution_confidence >= 0.5:
            return GraspSafetyDecision(
                GraspDiagnosis.POSSIBLE_OBSTRUCTION, RecommendedAction.SAFE_STOP,
                min(visual_confidence, execution_confidence), evidence,
            )
        return GraspSafetyDecision(
            GraspDiagnosis.MISSED_OR_EMPTY_GRASP, RecommendedAction.STOP_AND_REPLAN,
            visual_confidence, evidence,
        )
    if grasp.state is GraspControlState.MOVED_NOT_CONTROLLED:
        return GraspSafetyDecision(
            GraspDiagnosis.POSSIBLE_PUSH_OR_COLLISION,
            RecommendedAction.REQUEST_MORE_EVIDENCE, visual_confidence, evidence,
        )
    return GraspSafetyDecision(
        GraspDiagnosis.NORMAL_OR_PENDING, RecommendedAction.CONTINUE,
        visual_confidence, evidence,
    )
