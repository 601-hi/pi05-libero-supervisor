from vla_supervisor.events import RecommendedAction
from vla_supervisor.grasp_control_state import GraspControlResult, GraspControlState
from vla_supervisor.grasp_safety_fusion import GraspDiagnosis, fuse_grasp_safety


def result(state, valid=True):
    return GraspControlResult(state, None, 20, 0, valid)


def test_missed_grasp_replans_without_claiming_obstruction():
    decision = fuse_grasp_safety(
        result(GraspControlState.GRASP_NOT_ESTABLISHED), visual_confidence=.8,
        execution_mismatch=False, execution_confidence=.9,
    )
    assert decision.diagnosis is GraspDiagnosis.MISSED_OR_EMPTY_GRASP
    assert decision.action is RecommendedAction.STOP_AND_REPLAN


def test_independent_execution_mismatch_upgrades_to_possible_obstruction():
    decision = fuse_grasp_safety(
        result(GraspControlState.GRASP_NOT_ESTABLISHED), visual_confidence=.8,
        execution_mismatch=True, execution_confidence=.7,
    )
    assert decision.diagnosis is GraspDiagnosis.POSSIBLE_OBSTRUCTION
    assert decision.action is RecommendedAction.SAFE_STOP
    assert decision.confidence == .7


def test_unknown_requests_evidence_and_drop_replans():
    unknown = fuse_grasp_safety(
        result(GraspControlState.UNKNOWN, valid=False), visual_confidence=.2,
        execution_mismatch=True, execution_confidence=.9,
    )
    assert unknown.action is RecommendedAction.REQUEST_MORE_EVIDENCE
    dropped = fuse_grasp_safety(
        result(GraspControlState.CONTROL_LOST), visual_confidence=.75,
        execution_mismatch=False, execution_confidence=.1,
    )
    assert dropped.diagnosis is GraspDiagnosis.POSSIBLE_DROP
    assert dropped.action is RecommendedAction.STOP_AND_REPLAN


def test_motion_without_control_remains_selective():
    decision = fuse_grasp_safety(
        result(GraspControlState.MOVED_NOT_CONTROLLED), visual_confidence=.6,
        execution_mismatch=False, execution_confidence=.4,
    )
    assert decision.diagnosis is GraspDiagnosis.POSSIBLE_PUSH_OR_COLLISION
    assert decision.action is RecommendedAction.REQUEST_MORE_EVIDENCE
