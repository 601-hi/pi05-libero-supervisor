from cross_suite_generalization.build_causal_identity_holdout import build, annotation_template
from cross_suite_generalization.evaluate_causal_identity_holdout import evaluate


def sample_data():
    goals = ["pick up bowl", "put pot"]
    jobs = []
    for goal in goals:
        for episode in range(3):
            jobs.append({
                "suite": "suite-a", "goal_language": goal,
                "task_id_for_data_join_only": goals.index(goal), "episode_idx": episode,
                "sidecar": f"/{goal}/{episode}_{'failure' if episode == 2 else 'success'}.npz",
                "outcome_for_audit_only": "failure" if episode == 2 else "success",
            })
    pilot = {"episodes": [{"goal_language": goal} for goal in goals]}
    return {"jobs": jobs}, pilot


def test_holdout_excludes_exact_pilot_and_hides_outcomes():
    public, private = build(*sample_data(), wave_size=2)
    assert len(public["records"]) == 4
    assert len(private["records"]) == 4
    serialized_records = str(public["records"]).lower()
    assert "outcome" not in serialized_records
    assert "success" not in serialized_records
    assert "failure" not in serialized_records
    assert ".npz" not in serialized_records
    assert all(row["evaluation_wave"] in (1, 2) for row in public["records"])


def test_hash_split_is_deterministic():
    first, _ = build(*sample_data())
    second, _ = build(*sample_data())
    assert first == second


def test_evaluator_reports_rejection_and_false_lock():
    public, _ = build(*sample_data())
    annotations = annotation_template(public)
    for index, row in enumerate(annotations["records"]):
        row["identity_observable"] = "yes"
        row["manipulated_candidate_id"] = 3
        row["first_observable_frame"] = 5
    predictions = {"records": []}
    for index, row in enumerate(public["records"]):
        predictions["records"].append({
            "anonymous_id": row["anonymous_id"],
            "locked_candidate_id": (3, 9, None, 3)[index],
            "lock_frame": 8,
        })
    result = evaluate(public, annotations, predictions)
    assert result["correct_lock_rate_on_observable"] == 0.5
    assert result["false_lock_rate_all"] == 0.25
    assert result["unknown_rate_all"] == 0.25
    assert result["mean_delay_frames_when_correct"] == 3
