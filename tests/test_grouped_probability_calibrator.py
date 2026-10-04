import numpy as np
import pytest

from cross_suite_generalization.fit_grouped_probability_calibrator import fit_artifact


def synthetic():
    rng = np.random.default_rng(7)
    groups = np.repeat(np.arange(20), 5).astype(str)
    x = rng.normal(size=(100, 2))
    y = (x[:, 0] + .5*x[:, 1] > 0).astype(np.int64)
    waves = np.where(np.arange(100) % 2, "Wave1", "Wave2")
    return x, y, groups, waves


def test_grouped_calibrator_produces_oof_metrics_and_parameters():
    x, y, groups, waves = synthetic()
    result = fit_artifact(x, y, groups, waves,
                          feature_names=["a", "b"], source_hash="abc", folds=5)
    assert result["grouped_oof"]["auc"] > .9
    assert len(result["parameters"]["coefficients"]) == 2
    assert result["episode_groups"] == 20


def test_future_wave_is_rejected():
    x, y, groups, waves = synthetic()
    waves[0] = "Wave4"
    with pytest.raises(ValueError, match="only Wave1/Wave2"):
        fit_artifact(x, y, groups, waves,
                     feature_names=["a", "b"], source_hash="abc", folds=5)
