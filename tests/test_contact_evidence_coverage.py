from cross_suite_generalization.audit_contact_evidence_coverage import audit


def test_missing_probabilities_remain_missing_instead_of_being_imputed():
    identifier = "episode-a"
    result = audit(
        {identifier: {"background_confidence_sequence": [.8, .9]}},
        {identifier: {"events": [{"type": "attachment_established"}]}},
        {identifier: {"scored_steps": 2}},
    )
    row = result["records"][0]
    assert row["available_fields"]["execution_four_state"]
    assert row["available_fields"]["background_confidence"]
    assert not row["available_fields"]["attachment_probability"]
    assert not row["available_fields"]["independent_motion_probability"]
    assert not row["prospective_contact_replay_ready"]


def test_union_exposes_episode_alignment_gaps():
    result = audit(
        {"background-only": {"background_confidence_sequence": [.8]}},
        {"wrist-only": {"events": []}},
        {},
    )
    assert result["summary"]["episodes"] == 2
    assert result["summary"]["field_episode_coverage"]["background_confidence"] == .5
    assert result["summary"]["field_episode_coverage"]["execution_four_state"] == 0.0
