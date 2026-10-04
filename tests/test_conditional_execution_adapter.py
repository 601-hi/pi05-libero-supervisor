import numpy as np
import pytest

from vla_supervisor.conditional_execution import ConditionalDualExpertScorer


def test_gripper_fraction_uses_finger_separation():
    state = {"gripper_qpos": np.asarray([.04, -.04])}
    assert ConditionalDualExpertScorer._gripper_fraction(state) == pytest.approx(1.0)


def test_reset_clears_all_causal_state():
    scorer = object.__new__(ConditionalDualExpertScorer)
    scorer.window = 10
    scorer.reset()
    scorer.initial_eef = np.ones(3)
    scorer.previous_response[:] = 2
    scorer.previous_progress = 3
    scorer.robust_scores.extend([4, 5])
    scorer.reset()
    assert scorer.initial_eef is None
    assert np.all(scorer.previous_response == 0)
    assert scorer.previous_progress == 0
    assert len(scorer.robust_scores) == 0
