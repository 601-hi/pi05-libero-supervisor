import pytest

from vla_supervisor.wrist_physical_evidence import (
    WristCandidateScore,
    PlattCalibration,
    WristSelectionConfig,
    WristSelectionState,
    select_wrist_candidates,
)


def test_close_scores_return_set_valued_ambiguity() -> None:
    result = select_wrist_candidates([
        WristCandidateScore("a", .82, .9),
        WristCandidateScore("b", .76, .8),
        WristCandidateScore("c", .50, 1.0),
    ])
    assert result.state is WristSelectionState.AMBIGUOUS
    assert result.candidate_ids == ("a", "b")
    assert result.calibrated_attachment_probability is None


def test_clear_winner_is_unique_but_not_automatically_a_probability() -> None:
    result = select_wrist_candidates([
        WristCandidateScore("a", .90, .9),
        WristCandidateScore("b", .60, .9),
    ])
    assert result.state is WristSelectionState.UNIQUE
    assert result.candidate_ids == ("a",)
    assert result.runner_up_gap == pytest.approx(.30)
    assert result.calibrated_attachment_probability is None


def test_invisible_candidates_produce_explicit_abstention() -> None:
    result = select_wrist_candidates(
        [WristCandidateScore("a", .99, .2)],
        WristSelectionConfig(minimum_visibility=.5),
    )
    assert result.state is WristSelectionState.NO_CANDIDATE
    assert result.candidate_ids == ()


def test_calibration_is_exposed_only_for_unique_selection() -> None:
    calibration = PlattCalibration(mean=.5, scale=.2, intercept=-1.0, coefficient=2.0)
    unique = select_wrist_candidates([
        WristCandidateScore("a", .9, 1.0), WristCandidateScore("b", .5, 1.0)
    ], calibration=calibration)
    ambiguous = select_wrist_candidates([
        WristCandidateScore("a", .9, 1.0), WristCandidateScore("b", .85, 1.0)
    ], calibration=calibration)
    assert unique.calibrated_attachment_probability is not None
    assert unique.calibrated_attachment_probability > .5
    assert ambiguous.calibrated_attachment_probability is None
