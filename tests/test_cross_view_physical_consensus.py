from cross_suite_generalization.evaluate_cross_view_physical_consensus import consensus


def test_unique_intersection_confirms_only_shared_candidate() -> None:
    appearance = {"candidate_scores": [
        {"candidate_id": 1, "appearance": .90},
        {"candidate_id": 2, "appearance": .88},
        {"candidate_id": 3, "appearance": .50},
    ]}
    physical = [
        {"candidate_id": 1, "physical_fusion_score": .60},
        {"candidate_id": 2, "physical_fusion_score": .85},
        {"candidate_id": 3, "physical_fusion_score": .80},
    ]
    result = consensus(appearance, physical, appearance_margin=.03, physical_margin=.10)
    assert result["state"] == "unique"
    assert result["candidate_ids"] == [2]


def test_empty_intersection_is_conflict_not_forced_choice() -> None:
    appearance = {"candidate_scores": [{"candidate_id": 1, "appearance": .9}, {"candidate_id": 2, "appearance": .2}]}
    physical = [{"candidate_id": 1, "physical_fusion_score": .2}, {"candidate_id": 2, "physical_fusion_score": .9}]
    result = consensus(appearance, physical, appearance_margin=.03, physical_margin=.10)
    assert result["state"] == "conflict"
    assert result["candidate_ids"] == []
