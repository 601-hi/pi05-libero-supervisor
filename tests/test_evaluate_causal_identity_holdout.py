from cross_suite_generalization.evaluate_causal_identity_holdout import evaluate


def _public():
    return {"records": [{"anonymous_id": "x", "family": "rigid", "evaluation_wave": 3}]}


def _prediction(candidate_id=2):
    return {"records": [{"anonymous_id": "x", "locked_candidate_id": candidate_id, "lock_frame": 4}]}


def test_accepts_multi_candidate_mask_annotation():
    annotations = {
        "records": [{
            "anonymous_id": "x",
            "target_mask_observable": True,
            "acceptable_manipulated_candidate_ids": [2, 3],
        }]
    }
    result = evaluate(_public(), annotations, _prediction(3))
    assert result["records"][0]["correct_lock"] is True


def test_unobservable_lock_is_false_lock():
    annotations = {
        "records": [{
            "anonymous_id": "x",
            "target_mask_observable": False,
            "acceptable_manipulated_candidate_ids": [],
        }]
    }
    result = evaluate(_public(), annotations, _prediction(2))
    assert result["records"][0]["false_lock"] is True


def test_legacy_annotation_remains_supported():
    annotations = {
        "records": [{
            "anonymous_id": "x",
            "identity_observable": "yes",
            "manipulated_candidate_id": 2,
            "first_observable_frame": 1,
        }]
    }
    result = evaluate(_public(), annotations, _prediction(2))
    assert result["records"][0]["correct_lock"] is True
    assert result["records"][0]["delay_frames"] == 3
