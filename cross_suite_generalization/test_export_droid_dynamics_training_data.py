import unittest

import numpy as np

from export_droid_dynamics_training_data import split_for, wrap_angle_difference


class ExportDroidDynamicsTrainingDataTest(unittest.TestCase):
    def test_episode_split_is_deterministic(self):
        self.assertEqual(split_for("episode-a"), split_for("episode-a"))
        self.assertIn(split_for("episode-a"), {"train", "calibration"})

    def test_rotation_wrap_avoids_two_pi_jump(self):
        before = np.asarray([np.pi - 0.01])
        after = np.asarray([-np.pi + 0.02])
        np.testing.assert_allclose(wrap_angle_difference(after, before), [0.03], atol=1e-12)


if __name__ == "__main__":
    unittest.main()
