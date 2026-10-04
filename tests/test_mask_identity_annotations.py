from cross_suite_generalization.generate_mask_identity_annotation_template import build_template
from cross_suite_generalization.validate_mask_identity_annotations import validate


def _predictions():
    return {"records": [{"anonymous_id": "a", "num_candidates": 3}]}


def test_generated_template_is_intentionally_incomplete():
    template = build_template(_predictions())
    errors = validate(_predictions(), template)
    assert any("target_mask_observable" in error for error in errors)
    assert any("confidence" in error for error in errors)


def test_completed_annotation_is_valid():
    annotation = {"records": [{
        "anonymous_id": "a",
        "acceptable_target_mask_ids": [1, 3],
        "target_mask_observable": True,
        "confidence": "high",
        "notes": "two nested masks both cover the target",
    }]}
    assert validate(_predictions(), annotation) == []


def test_rejects_wrong_order_duplicates_and_out_of_range_ids():
    predictions = {"records": [
        {"anonymous_id": "a", "num_candidates": 2},
        {"anonymous_id": "b", "num_candidates": 1},
    ]}
    annotation = {"records": [
        {"anonymous_id": "b", "acceptable_target_mask_ids": [2, 2], "target_mask_observable": True, "confidence": "high"},
        {"anonymous_id": "a", "acceptable_target_mask_ids": [], "target_mask_observable": False, "confidence": "medium"},
    ]}
    errors = validate(predictions, annotation)
    assert any("order" in error for error in errors)
    assert any("duplicate target" in error for error in errors)
    assert any("invalid IDs" in error for error in errors)
