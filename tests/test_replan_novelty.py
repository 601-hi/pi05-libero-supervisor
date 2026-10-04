import numpy as np

from vla_supervisor.replan_novelty import ReplanNoveltyGate


def chunk():
    return np.asarray([
        [.3, .1, 0, 0, 0, .1, -1],
        [.2, .1, 0, 0, 0, .1, -1],
        [.1, .2, 0, 0, 0, .1, -1],
        [0, .3, 0, 0, 0, .1, 1],
        [0, .2, 0, 0, 0, .1, 1],
    ], dtype=float)


def test_identical_and_small_noise_are_repeated():
    gate = ReplanNoveltyGate()
    old = chunk()
    assert gate.compare(old, old.copy()).state == "repeated"
    new = old.copy(); new[:, :6] *= .95
    assert gate.compare(old, new).state == "repeated"


def test_direction_or_gripper_change_is_material():
    gate = ReplanNoveltyGate()
    old = chunk()
    reversed_motion = old.copy(); reversed_motion[:, :6] *= -1
    assert gate.compare(old, reversed_motion).state == "changed"
    changed_grasp = old.copy(); changed_grasp[:, 6] *= -1
    assert gate.compare(old, changed_grasp).state == "changed"


def test_missing_or_short_reference_fails_open_for_first_version():
    gate = ReplanNoveltyGate()
    assert gate.compare(None, chunk()).state == "insufficient_evidence"
    assert gate.compare(chunk()[:2], chunk()[:2]).accept
