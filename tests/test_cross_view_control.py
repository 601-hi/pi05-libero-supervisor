import pytest

from vla_supervisor.cross_view_control import CrossViewEvidence, CrossViewState, cross_validate_views


def test_two_views_confirm_attachment():
    result = cross_validate_views(CrossViewEvidence(True, True, 0.9, 0.8))
    assert result.state is CrossViewState.CONFIRMED_ATTACHED
    assert not result.request_more_evidence


def test_wrist_motion_without_world_motion_is_conflict():
    result = cross_validate_views(CrossViewEvidence(False, True, 0.9, 0.8))
    assert result.state is CrossViewState.CONFLICT
    assert result.request_more_evidence


def test_low_confidence_wrist_does_not_veto_fixed_view():
    result = cross_validate_views(CrossViewEvidence(True, False, 0.9, 0.2))
    assert result.state is CrossViewState.FIXED_ONLY_MOTION
    assert result.request_more_evidence


def test_both_invalid_views_return_unknown():
    result = cross_validate_views(CrossViewEvidence(True, True, 0.2, 0.2))
    assert result.state is CrossViewState.UNKNOWN


def test_invalid_confidence_is_rejected():
    with pytest.raises(ValueError):
        cross_validate_views(CrossViewEvidence(True, True, 1.2, 0.8))

