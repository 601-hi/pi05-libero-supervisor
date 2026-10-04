import numpy as np
import pytest

from vla_supervisor.candidate_relation_graph import build_candidate_relation_graph


def test_duplicate_masks_are_equivalent_not_parts():
    a = np.zeros((20, 20), bool); a[4:12, 4:12] = 1
    b = a.copy(); b[11, 11] = 0
    graph = build_candidate_relation_graph({1: a, 2: b})
    assert graph.equivalents(1) == frozenset({1, 2})
    assert not graph.whole_candidates_for(1)


def test_small_contained_mask_is_part_of_large_candidate():
    part = np.zeros((20, 20), bool); part[7:10, 7:10] = 1
    whole = np.zeros((20, 20), bool); whole[4:14, 4:14] = 1
    graph = build_candidate_relation_graph({"handle": part, "pot": whole})
    assert graph.whole_candidates_for("handle") == frozenset({"pot"})


def test_shape_mismatch_is_rejected():
    with pytest.raises(ValueError):
        build_candidate_relation_graph({1: np.zeros((4, 4)), 2: np.zeros((5, 5))})


def test_touching_nonoverlapping_parts_are_adjacent_but_not_equivalent():
    body = np.zeros((20, 20), bool); body[5:12, 5:12] = 1
    handle = np.zeros((20, 20), bool); handle[7:10, 12:15] = 1
    graph = build_candidate_relation_graph({"body": body, "handle": handle})
    assert graph.adjacent_candidates("body", maximum_distance_px=1.0) == frozenset({"handle"})
    assert graph.equivalents("body") == frozenset({"body"})
