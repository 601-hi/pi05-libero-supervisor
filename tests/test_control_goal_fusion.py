import numpy as np

from vla_supervisor.candidate_relation_graph import build_candidate_relation_graph
from vla_supervisor.control_goal_fusion import fuse_control_and_goal


def graph_for_masks():
    handle = np.zeros((20, 20), bool); handle[7:10, 7:10] = 1
    pot = np.zeros((20, 20), bool); pot[4:14, 4:14] = 1
    bowl = np.zeros((20, 20), bool); bowl[15:19, 1:5] = 1
    return build_candidate_relation_graph({"handle": handle, "pot": pot, "bowl": bowl})


def test_control_without_semantics_does_not_claim_task_success():
    decision = fuse_control_and_goal(
        controlled_candidate_id="bowl", semantic_scores={}, relation_graph=graph_for_masks()
    )
    assert decision.state == "ambiguous"


def test_wrong_object_control_remains_explicit():
    decision = fuse_control_and_goal(
        controlled_candidate_id="bowl",
        semantic_scores={"pot": 0.9, "bowl": 0.1},
        relation_graph=graph_for_masks(),
    )
    assert decision.state == "wrong_object_control"


def test_semantically_supported_part_can_lift_to_whole():
    decision = fuse_control_and_goal(
        controlled_candidate_id="handle",
        semantic_scores={"pot": 0.9, "bowl": 0.1},
        relation_graph=graph_for_masks(),
    )
    assert decision.state == "target_controlled"
    assert decision.target_candidate_id == "pot"


def test_geometry_cannot_lift_part_when_semantics_are_ambiguous():
    decision = fuse_control_and_goal(
        controlled_candidate_id="handle",
        semantic_scores={"pot": 0.55, "bowl": 0.52},
        relation_graph=graph_for_masks(),
    )
    assert decision.state == "ambiguous"

