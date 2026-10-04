import pytest

from vla_supervisor.adaptive_budget import AdaptivePhaseBudget


def governor(**kwargs):
    return AdaptivePhaseBudget(
        soft_limit=10, hard_limit=30, extension_chunk=5, window=5,
        minimum_net_progress_m=0.0005,
        minimum_slope_m_per_step=0.00001, **kwargs,
    )


def test_extends_only_when_recent_error_is_converging():
    item = governor()
    for error in [0.0100, 0.0098, 0.0096, 0.0093, 0.0091]:
        item.observe(error, target_id=4)
    decision = item.decide(10)
    assert decision.allow and decision.extended
    assert decision.effective_limit == 15


@pytest.mark.parametrize("errors", [
    [0.0100, 0.0101, 0.0102, 0.0103, 0.0104],
    [0.0100, 0.0099, 0.0101, 0.0099, 0.0100],
])
def test_rejects_divergence_or_plateau(errors):
    item = governor()
    for error in errors:
        item.observe(error, target_id=4)
    decision = item.decide(10)
    assert not decision.allow
    assert decision.reason == "not_converging"


def test_target_change_resets_incomparable_replay_errors():
    item = governor()
    for error in [0.0100, 0.0098, 0.0096, 0.0093, 0.0091]:
        item.observe(error, target_id=4)
    item.observe(0.0200, target_id=3)
    decision = item.decide(10)
    assert not decision.allow
    assert decision.reason == "insufficient_progress_history"


def test_hard_limit_cannot_be_extended():
    item = governor()
    for error in [0.0100, 0.0098, 0.0096, 0.0093, 0.0091]:
        item.observe(error, target_id=4)
    assert item.decide(10).extended
    assert not item.decide(30).allow
    assert item.decide(30).reason == "hard_limit"


def test_replay_extension_requires_reference_advance():
    item = governor(require_target_advance=True)
    for error in [0.0100, 0.0098, 0.0096, 0.0093, 0.0091]:
        item.observe(error, target_id=4)
    assert item.decide(10).reason == "no_target_advance"

    item.reset()
    item.observe(0.0120, target_id=5)
    for error in [0.0100, 0.0098, 0.0096, 0.0093, 0.0091]:
        item.observe(error, target_id=4)
    decision = item.decide(10)
    assert decision.allow and decision.extended

