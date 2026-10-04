import math
import numpy as np

from cross_suite_generalization.analyze_fixed_gripper_changepoint import log_ratio, median_speed


def test_relative_speed_drop_is_positive_when_motion_settles() -> None:
    assert log_ratio(4.0, .2) > 2.0


def test_median_speed_ignores_missing_pairs() -> None:
    track = np.asarray([[0., 0.], [1., 0.], [math.nan, math.nan], [4., 0.], [5., 0.]])
    assert median_speed(track, 0, 5) == 1.0
