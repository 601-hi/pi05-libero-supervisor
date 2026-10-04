from cross_suite_generalization.validate_contact_annotations import validate


def valid_payload():
    index = {"records": [{"anonymous_id": "x", "frames": 20, "close_frames": [5]}]}
    annotations = {
        "annotation_pass": "A_physical_blind",
        "records": [{
            "anonymous_id": "x",
            "annotations": [{
                "close_frame": 5,
                "contact_onset_frame": 7,
                "contact_visibility": "visible",
                "contact_entity_type": "movable_object",
                "independent_motion_start": 8,
                "independent_motion_end": 12,
                "control_start": 9,
                "control_end": 12,
                "physical_mechanism": "successful_pickup",
                "evidence_quality": "high",
                "notes": "",
            }],
        }],
    }
    return annotations, index


def test_valid_annotation_passes():
    annotations, index = valid_payload()
    assert validate(annotations, index) == []


def test_forbidden_language_and_bad_interval_fail():
    annotations, index = valid_payload()
    annotations["records"][0]["goal_language"] = "leak"
    annotations["records"][0]["annotations"][0]["control_start"] = 15
    errors = validate(annotations, index)
    assert any("forbidden" in error for error in errors)
    assert any("exceeds" in error for error in errors)
