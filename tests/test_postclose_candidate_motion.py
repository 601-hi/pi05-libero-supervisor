from cross_suite_generalization.audit_postclose_candidate_motion import co_motion_groups, summarize_candidate


def test_sustained_motion_exceeds_candidate_specific_noise() -> None:
    sequence = [[10.0 + 0.1 * (index % 2), 20.0] for index in range(11)]
    sequence += [[14.0 + 4.0 * index, 20.0] for index in range(10)]
    result = summarize_candidate(sequence, close=10, pre=10, post=10)
    assert result["adaptive_motion_threshold_px"] == 3.0
    assert result["motion_onset_frame"] == 11
    assert result["sustained_motion"] is True


def test_tracker_jitter_is_not_called_sustained_motion() -> None:
    sequence = [[10.0 + 0.2 * (index % 2), 20.0] for index in range(21)]
    result = summarize_candidate(sequence, close=10, pre=10, post=10)
    assert result["motion_onset_frame"] is None
    assert result["sustained_motion"] is False


def test_missing_close_point_produces_unknown_motion() -> None:
    sequence = [[10.0, 20.0] for _ in range(21)]
    sequence[10] = None
    result = summarize_candidate(sequence, close=10, pre=10, post=10)
    assert result["postclose_net_displacement_px"] is None
    assert result["motion_onset_frame"] is None


def test_rigidly_moving_candidates_form_one_group() -> None:
    sequences = {
        "1": [[float(index), 0.0] for index in range(12)],
        "2": [[float(index) + 20.0, 10.0] for index in range(12)],
        "3": [[2.0 * float(index), 50.0] for index in range(12)],
    }
    groups = co_motion_groups(sequences, close=2, post=9, sustained_ids={"1", "2", "3"})
    assert ["1", "2"] in groups
    assert ["3"] in groups
