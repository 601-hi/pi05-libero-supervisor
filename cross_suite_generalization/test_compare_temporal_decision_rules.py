import numpy as np

from cross_suite_generalization.compare_temporal_decision_rules import Rule, conformal_upper, window_statistic


def test_vote_statistic_is_kth_largest():
    scores = np.array([1.0, 9.0, 3.0, 2.0])
    assert np.allclose(window_statistic(scores, Rule("vote", 2, 3)), [3.0, 3.0])


def test_mean_is_causal_full_window():
    assert np.allclose(window_statistic(np.array([1.0, 2.0, 6.0]), Rule("mean", 0, 2)), [1.5, 4.0])


def test_conformal_upper_uses_finite_sample_rank():
    threshold, rank = conformal_upper(np.arange(1.0, 10.0), 0.10)
    assert rank == 9
    assert threshold == 9.0
