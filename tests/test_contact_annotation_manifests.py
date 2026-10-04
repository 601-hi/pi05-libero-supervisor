from cross_suite_generalization.build_contact_annotation_manifests import build


def test_physical_pass_is_blind_to_language_and_semantic_pass_is_deferred():
    records = [{
        "anonymous_id": "x", "development_wave": "Wave1", "frame_count": 50,
        "close_frames": [20], "goal_language": "pick up the black bowl",
    }]
    physical, semantic = build(records)
    assert "goal_language" not in physical["records"][0]
    assert physical["records"][0]["annotations"][0]["close_frame"] == 20
    assert physical["records"][0]["annotations"][0]["physical_mechanism"] is None
    assert semantic["records"][0]["goal_language"] == "pick up the black bowl"
    assert semantic["records"][0]["physical_annotation_reference"] is None


def test_duplicate_ids_are_rejected():
    row = {"anonymous_id": "x", "development_wave": "Wave1", "frame_count": 1,
           "close_frames": [], "goal_language": "task"}
    try:
        build([row, row])
    except ValueError as error:
        assert "overlap" in str(error)
    else:
        raise AssertionError("duplicate anonymous ids must fail")
