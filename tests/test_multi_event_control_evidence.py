from vla_supervisor.multi_event_control_evidence import ControlEvidenceState, classify_control_evidence


def classify(**overrides):
    values = dict(wrist_state="unique", wrist_passes_gate=False,
                  fixed_has_sustained_motion=False, best_sustained_change=None)
    values.update(overrides)
    return classify_control_evidence(**values)


def test_possible_control_never_establishes_without_identity_bridge():
    result = classify(wrist_passes_gate=True, fixed_has_sustained_motion=True,
                      best_sustained_change=.4)
    assert result.state == ControlEvidenceState.POSSIBLE_NEW_CONTROL
    assert result.establish_control is False


def test_preexisting_robot_motion_is_not_control():
    result = classify(wrist_passes_gate=True, fixed_has_sustained_motion=True,
                      best_sustained_change=-.1)
    assert result.state == ControlEvidenceState.PREEXISTING_OR_ROBOT_MOTION
    assert result.establish_control is False


def test_right_censored_event_requests_more_frames():
    result = classify(wrist_state="insufficient_post_window", fixed_has_sustained_motion=True,
                      best_sustained_change=.8)
    assert result.state == ControlEvidenceState.INSUFFICIENT_WINDOW


def test_single_view_evidence_remains_single_view():
    assert classify(wrist_passes_gate=True).state == ControlEvidenceState.WRIST_CANDIDATE_ONLY
    assert classify(fixed_has_sustained_motion=True, best_sustained_change=.5).state == ControlEvidenceState.FIXED_MOTION_ONLY


def test_evidence_from_different_candidates_is_rejected():
    result = classify(
        wrist_passes_gate=True,
        fixed_has_sustained_motion=True,
        best_sustained_change=.4,
        same_candidate_semantic_alignment_count=0,
    )
    assert result.state == ControlEvidenceState.EVIDENCE_SPLIT_ACROSS_CANDIDATES
    assert result.establish_control is False
