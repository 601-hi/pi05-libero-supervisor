import numpy as np

from cross_suite_generalization.mechanism_conditional_calibration import (
    apply_normal_standardizer, assign_regime, fit_normal_standardizer,
)


def test_regimes_use_only_pre_action_inputs_and_priority():
    result = assign_regime(
        [0.04, 0.04, 0.01, 0.01], [0.1, 0.5, 0.1, 0.5],
        [0.5, 0.5, 0.5, 0.5], [0.0, 0.0, 0.0, 0.0],
    )
    assert result.tolist() == [0, 1, 2, 3]


def test_near_limit_and_gripper_transition_enter_special():
    result = assign_regime([0.01, 0.04], [0.5, 0.5], [0.05, 0.5], [0.0, 0.01])
    assert result.tolist() == [0, 0]


def test_fit_uses_group_median_iqr_and_has_sparse_fallback():
    score = np.r_[np.arange(20.0), [100.0]]
    regime = np.r_[np.zeros(20, dtype=int), [1]]
    params = fit_normal_standardizer(score, regime, min_group_samples=20)
    assert params[0][3] is False
    assert params[1][3] is True
    transformed = apply_normal_standardizer(score, regime, params)
    assert np.isfinite(transformed).all()
