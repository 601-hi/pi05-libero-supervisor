import pytest

from vla_supervisor.adaptive_motion_baseline import AdaptiveMotionBaseline


def test_baseline_scores_drop_without_contaminating_reference():
    baseline = AdaptiveMotionBaseline(window=10, minimum_samples=4, minimum_scale=.001)
    for value in (0.20, 0.22, 0.19, 0.21):
        baseline.observe(value, background_confidence=.9, allow_reference_update=True)
    before = list(baseline.values)
    result = baseline.observe(.01, background_confidence=.9, allow_reference_update=False)
    assert result.ready
    assert result.motion_drop_z > 0
    assert not result.update_accepted
    assert list(baseline.values) == before


def test_current_sample_is_scored_before_reference_update():
    baseline = AdaptiveMotionBaseline(window=6, minimum_samples=2, minimum_scale=.001)
    baseline.observe(.2, background_confidence=1, allow_reference_update=True)
    baseline.observe(.2, background_confidence=1, allow_reference_update=True)
    result = baseline.observe(2.0, background_confidence=1, allow_reference_update=True)
    assert result.motion_excess_z > 0
    assert result.samples == 3


def test_low_background_confidence_refuses_reference_update():
    baseline = AdaptiveMotionBaseline(minimum_background_confidence=.5)
    result = baseline.observe(.2, background_confidence=.49, allow_reference_update=True)
    assert result.status == "reference_update_refused"
    assert result.samples == 0


@pytest.mark.parametrize("value", [None, -1, float("nan")])
def test_invalid_measurements_are_explicitly_rejected(value):
    result = AdaptiveMotionBaseline().observe(
        value, background_confidence=1, allow_reference_update=True
    )
    assert result.status == "invalid_measurement"
    assert not result.update_accepted
