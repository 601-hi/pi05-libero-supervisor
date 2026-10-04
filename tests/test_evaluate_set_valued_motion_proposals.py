from cross_suite_generalization.evaluate_set_valued_motion_proposals import evaluate, proposal_ids


def test_proposal_is_union_of_lock_and_largest_displacement() -> None:
    replay = {"first_lock": {"candidate_id": 2}}
    feature = {"candidates": {"1": {"maximum_net_displacement": 0.7}, "2": {"maximum_net_displacement": 0.3}}}
    assert proposal_ids(replay, feature) == {1, 2}


def test_duplicate_evidence_keeps_single_candidate() -> None:
    replay = {"first_lock": {"candidate_id": 1}}
    feature = {"candidates": {"1": {"maximum_net_displacement": 0.7}, "2": {"maximum_net_displacement": 0.3}}}
    assert proposal_ids(replay, feature) == {1}


def test_evaluation_excludes_unobservable_from_recall_denominator() -> None:
    replay = [{"anonymous_id": "a", "first_lock": {"candidate_id": 2}}, {"anonymous_id": "b", "first_lock": None}]
    features = [
        {"anonymous_id": "a", "candidates": {"1": {"maximum_net_displacement": 0.8}, "2": {"maximum_net_displacement": 0.2}}},
        {"anonymous_id": "b", "candidates": {"3": {"maximum_net_displacement": 0.5}}},
    ]
    annotations = [
        {"anonymous_id": "a", "target_mask_observable": True, "acceptable_target_mask_ids": [2]},
        {"anonymous_id": "b", "target_mask_observable": False, "acceptable_target_mask_ids": []},
    ]
    result = evaluate(replay, features, annotations)["summary"]
    assert result["observable_episodes"] == 1
    assert result["hits"] == 1
    assert result["recall"] == 1.0
    assert result["average_proposal_size"] == 2.0
