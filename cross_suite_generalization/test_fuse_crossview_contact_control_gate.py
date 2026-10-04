from cross_suite_generalization.fuse_crossview_contact_control_gate import evidence_states


def test_no_fixed_motion_has_priority():
    assert evidence_states(0, 1, 0.99, 0.7) == (
        "no_motion", "unique_high_confidence", "no_fixed_motion_evidence"
    )


def test_single_fixed_and_unique_high_is_possible_control():
    assert evidence_states(1, 1, 0.8, 0.7) == (
        "single_motion_pattern", "unique_high_confidence", "possible_control"
    )


def test_multiple_fixed_patterns_remain_ambiguous():
    assert evidence_states(3, 1, 0.8, 0.7) == (
        "multiple_motion_patterns", "unique_high_confidence", "possible_control_crossview_ambiguous"
    )


def test_wrist_abstention_states():
    assert evidence_states(1, 0, None, 0.7)[2] == "wrist_no_candidate"
    assert evidence_states(1, 2, None, 0.7)[2] == "wrist_ambiguous"
    assert evidence_states(1, 1, 0.69, 0.7)[2] == "unique_wrist_low_confidence"
