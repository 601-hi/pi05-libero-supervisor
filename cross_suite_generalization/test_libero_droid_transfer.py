import unittest
import numpy as np

from libero_droid_transfer import gripper_open_fraction, intended_velocity, quaternion_wxyz_to_euler, wrapped_delta


class TransferPrimitiveTests(unittest.TestCase):
    def test_identity_quaternion(self):
        np.testing.assert_allclose(quaternion_wxyz_to_euler([1, 0, 0, 0]), np.zeros(3), atol=1e-12)

    def test_controller_spec_conversion(self):
        got = intended_velocity([1, -1, 0.5, 0.2, -0.2, 0.1, 1])
        np.testing.assert_allclose(got, [1, -1, 0.5, 2, -2, 1])

    def test_clips_controller_action(self):
        got = intended_velocity([2, -2, 0, 2, -2, 0, 1])
        np.testing.assert_allclose(got, [1, -1, 0, 10, -10, 0])

    def test_wrap_crossing(self):
        np.testing.assert_allclose(wrapped_delta([-np.pi + 0.1], [np.pi - 0.1]), [0.2])

    def test_gripper_fraction(self):
        np.testing.assert_allclose(gripper_open_fraction([0.04, -0.04]), [1.0])
        np.testing.assert_allclose(gripper_open_fraction([0.0, 0.0]), [0.0])


if __name__ == "__main__":
    unittest.main()
