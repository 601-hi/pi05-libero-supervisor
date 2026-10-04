from vla_supervisor.causal_object_identity import CausalMotionIdentitySelector


def test_static_candidates_remain_unknown():
    selector = CausalMotionIdentitySelector((100, 100), minimum_motion=0.05)
    for _ in range(10):
        state = selector.update({1: (10, 10), 2: (80, 80)})
    assert state.locked_object_id is None


def test_sustained_separated_motion_locks_causally():
    selector = CausalMotionIdentitySelector(
        (100, 100), minimum_motion=0.05, minimum_margin=0.02, confirmation_frames=3
    )
    selector.update({1: (10, 10), 2: (80, 80)})
    assert selector.update({1: (20, 10), 2: (80, 80)}).locked_object_id is None
    assert selector.update({1: (21, 10), 2: (80, 80)}).locked_object_id is None
    assert selector.update({1: (22, 10), 2: (80, 80)}).locked_object_id == 1


def test_short_spike_does_not_lock_and_lock_is_sticky():
    selector = CausalMotionIdentitySelector(
        (100, 100), minimum_motion=0.05, minimum_margin=0.02, confirmation_frames=2
    )
    selector.update({1: (10, 10), 2: (80, 80)})
    selector.update({1: (20, 10), 2: (80, 80)})
    # Candidate 1's one-frame excursion remains historical evidence, so candidate 2
    # must exceed it by the configured margin for two consecutive observations.
    assert selector.update({1: (10, 10), 2: (105, 80)}).locked_object_id is None
    assert selector.update({1: (10, 10), 2: (106, 80)}).locked_object_id == 2
    assert selector.update({1: (60, 10), 2: (106, 80)}).locked_object_id == 2


def test_invalid_configuration_is_rejected():
    import pytest

    with pytest.raises(ValueError):
        CausalMotionIdentitySelector((0, 100))
    with pytest.raises(ValueError):
        CausalMotionIdentitySelector((100, 100), confirmation_frames=0)
