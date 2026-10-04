import math
import unittest

import numpy as np

from kinematic_safety import compute_kinematic_safety


class KinematicSafetyTest(unittest.TestCase):
    def test_well_conditioned_center_configuration(self):
        m = compute_kinematic_safety(np.eye(3), [0, 0, 0], [[-1, 1]] * 3)
        self.assertAlmostEqual(m.sigma_min, 1.0)
        self.assertAlmostEqual(m.condition_number, 1.0)
        self.assertAlmostEqual(m.translation_sigma_min, 1.0)
        self.assertTrue(math.isnan(m.rotation_sigma_min))
        self.assertAlmostEqual(m.joint_limit_margin_fraction, 0.5)
        self.assertFalse(m.near_singular)
        self.assertFalse(m.near_joint_limit)

    def test_singular_jacobian(self):
        j = np.diag([1.0, 0.2, 1e-5])
        m = compute_kinematic_safety(j, [0, 0, 0], [[-1, 1]] * 3)
        self.assertTrue(m.near_singular)
        self.assertGreater(m.condition_number, 1000)

    def test_joint_limit_and_unbounded_joint(self):
        m = compute_kinematic_safety(
            np.eye(3), [0.99, 0.0, 10.0], [[-1, 1], [-2, 2], [0, 0]]
        )
        self.assertTrue(m.near_joint_limit)
        self.assertAlmostEqual(m.joint_limit_margin_fraction, 0.005)

    def test_invalid_values_fail_safe(self):
        m = compute_kinematic_safety([[1, 0], [0, math.nan]], [0, 0], [[-1, 1]] * 2)
        self.assertTrue(m.invalid)
        self.assertTrue(m.near_singular)

    def test_spatial_jacobian_reports_translation_and_rotation_separately(self):
        translation = np.diag([2.0, 1.0, 0.5])
        rotation = np.diag([3.0, 1.5, 0.75])
        j = np.c_[np.vstack([translation, rotation]), np.zeros((6, 3))]
        m = compute_kinematic_safety(j, [0] * 6, [[-1, 1]] * 6)
        self.assertAlmostEqual(m.translation_sigma_min, 0.5)
        self.assertAlmostEqual(m.translation_condition_number, 4.0)
        self.assertAlmostEqual(m.rotation_sigma_min, 0.75)
        self.assertAlmostEqual(m.rotation_condition_number, 4.0)


if __name__ == "__main__":
    unittest.main()
