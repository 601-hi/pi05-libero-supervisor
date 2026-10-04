import numpy as np

from cross_suite_generalization.evaluate_background_health import finite


def test_finite_drops_missing_and_nonfinite_values() -> None:
    result = finite([None, 0.2, float("nan"), 0.7, float("inf")])
    np.testing.assert_allclose(result, [0.2, 0.7])
