from cross_suite_generalization.run_causal_identity_holdout_gpu import selected_jobs


def test_selected_jobs_joins_private_paths_without_exposing_labels_to_public_rows():
    public = {"records": [
        {"anonymous_id": "a", "family": "rigid_object_transport", "evaluation_wave": 1},
        {"anonymous_id": "b", "family": "rigid_object_transport", "evaluation_wave": 2},
    ]}
    private = {"records": [
        {"anonymous_id": "a", "original_sidecar": "/private/failure.npz"},
        {"anonymous_id": "b", "original_sidecar": "/private/success.npz"},
    ]}
    jobs = selected_jobs(public, private, 1, "rigid_object_transport")
    assert [row["anonymous_id"] for row in jobs] == ["a"]
    assert jobs[0]["sidecar"] == "/private/failure.npz"
