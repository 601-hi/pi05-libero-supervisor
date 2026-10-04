import unittest

import numpy as np

from gripper_intervention import apply_gripper_intervention


class GripperInterventionTest(unittest.TestCase):
    def test_default_off_is_bitwise_identical(self):
        action = np.array([0.1, -0.2, 0.3, 0, 0, 0, 0.4])
        result = apply_gripper_intervention(
            action, active=False, mode="none", intended_gripper_history=[]
        )
        np.testing.assert_array_equal(result.executed_action, action)
        self.assertFalse(result.applied)

    def test_force_open_changes_only_gripper(self):
        action = np.arange(7, dtype=float)
        result = apply_gripper_intervention(
            action, active=True, mode="force_open", intended_gripper_history=[]
        )
        np.testing.assert_array_equal(result.executed_action[:-1], action[:-1])
        self.assertEqual(result.executed_action[-1], -1.0)
        self.assertTrue(result.applied)

    def test_force_closed_changes_only_gripper(self):
        action = np.zeros(7)
        result = apply_gripper_intervention(
            action, active=True, mode="force_closed", intended_gripper_history=[]
        )
        np.testing.assert_array_equal(result.executed_action[:-1], action[:-1])
        self.assertEqual(result.executed_action[-1], 1.0)

    def test_delay_is_causal(self):
        action = np.zeros(7)
        action[-1] = 0.8
        result = apply_gripper_intervention(
            action,
            active=True,
            mode="delay",
            intended_gripper_history=[-1.0, -0.5, 0.25],
            delay_steps=2,
        )
        self.assertEqual(result.executed_action[-1], -0.5)
        self.assertEqual(result.source_history_offset, 2)

    def test_delay_without_history_does_not_claim_application(self):
        result = apply_gripper_intervention(
            np.zeros(7),
            active=True,
            mode="delay",
            intended_gripper_history=[-1.0],
            delay_steps=3,
        )
        self.assertFalse(result.applied)


if __name__ == "__main__":
    unittest.main()
