from cross_suite_generalization.compare_fixed_proposal_rankers import evaluate


def test_rankers_use_candidate_physics_inside_proposal():
    features = [{
        "anonymous_id": "x",
        "candidates": {
            "1": {"maximum_net_displacement": 0.1, "events": [{"post40_max_displacement": 0.01}]},
            "2": {"maximum_net_displacement": 0.2, "events": [{"post40_max_displacement": 0.04}]},
        },
    }]
    proposals = [{"anonymous_id": "x", "proposal_ids": [1, 2]}]
    annotations = [{"anonymous_id": "x", "target_mask_observable": True, "acceptable_target_mask_ids": [2]}]
    result = evaluate(features, proposals, annotations)
    assert result["summary"]["proposal_contains_truth"] == 1
    assert result["records"][0]["selected_ids"]["maximum_net_displacement"] == 2
    assert result["records"][0]["selected_ids"]["post_close_max_displacement"] == 2
