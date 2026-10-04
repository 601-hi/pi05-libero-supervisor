from vla_supervisor.relation_candidate_selector import select_source_consistent_object


def prediction(role, box, score=0.8):
    return {"role": role, "box_xyxy": box, "score": score}


def test_on_selects_small_object_near_support_not_support_sized_false_box():
    rows = [
        prediction("object_category", [10, 10, 20, 20]),
        prediction("object_category", [80, 50, 140, 110], 0.95),
        prediction("source_0", [5, 15, 30, 30]),
    ]
    result = select_source_consistent_object(rows, "on", (224, 224))
    assert result.object_prediction_index == 0


def test_between_selects_candidate_between_two_references():
    rows = [
        prediction("object_category", [100, 95, 110, 105]),
        prediction("object_category", [180, 180, 195, 195], 0.95),
        prediction("source_0", [40, 90, 60, 110]),
        prediction("source_1", [150, 90, 170, 110]),
    ]
    result = select_source_consistent_object(rows, "between", (224, 224))
    assert result.object_prediction_index == 0


def test_missing_source_returns_ambiguous_rejection():
    result = select_source_consistent_object(
        [prediction("object_category", [0, 0, 10, 10])], "in", (224, 224)
    )
    assert result.ambiguous
    assert result.object_prediction_index is None
