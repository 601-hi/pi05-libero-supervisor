import numpy as np

from cross_suite_generalization.fit_wrist_physical_calibrator import fit_logistic, sigmoid


def test_logistic_calibrator_orders_separable_scores() -> None:
    x = np.asarray([-2.0, -1.0, 1.0, 2.0])
    y = np.asarray([0, 0, 1, 1])
    weights = fit_logistic(x, y, l2=.1)
    probability = sigmoid(weights[0] + weights[1] * x)
    assert weights[1] > 0
    assert probability[0] < probability[-1]
    assert np.all((probability > 0) & (probability < 1))
