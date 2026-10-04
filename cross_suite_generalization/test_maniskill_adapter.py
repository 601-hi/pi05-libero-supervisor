import unittest

import numpy as np

from maniskill_adapter import ManiSkillConfig, adapt_trajectory


VERIFIED = ManiSkillConfig(
    control_mode="pd_joint_delta_pos",
    control_frequency_hz=20.0,
    robot_model="panda",
    command_semantics_verified=True,
    units_verified=True,
)


class ManiSkillAdapterTest(unittest.TestCase):
    def test_t_actions_require_t_plus_one_measured_states(self):
        rows = adapt_trajectory(
            np.zeros((2, 8)),
            {"joint_position": np.zeros((3, 9))},
            episode_id="trajectory-1",
            config=VERIFIED,
        )
        self.assertEqual(len(rows), 2)
        self.assertAlmostEqual(rows[0].delta_t_seconds, 0.05)
        self.assertEqual(rows[0].command_semantics.mode, "joint_delta_position")

    def test_state_length_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, r"T\+1"):
            adapt_trajectory(
                np.zeros((2, 8)),
                {"joint_position": np.zeros((2, 9))},
                episode_id="trajectory-1",
                config=VERIFIED,
            )

    def test_unknown_control_mode_is_rejected(self):
        config = ManiSkillConfig("mystery", 20.0, "panda", True, True)
        with self.assertRaisesRegex(ValueError, "unsupported or ambiguous"):
            adapt_trajectory(
                np.zeros((1, 8)),
                {"joint_position": np.zeros((2, 9))},
                episode_id="trajectory-1",
                config=config,
            )

    def test_unverified_config_is_rejected(self):
        config = ManiSkillConfig("pd_joint_delta_pos", 20.0, "panda")
        with self.assertRaisesRegex(ValueError, "command_semantics_verified|units_verified"):
            adapt_trajectory(
                np.zeros((1, 8)),
                {"joint_position": np.zeros((2, 9))},
                episode_id="trajectory-1",
                config=config,
            )

    def test_privileged_state_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "object_pose"):
            adapt_trajectory(
                np.zeros((1, 8)),
                {"object_pose": np.zeros((2, 7))},
                episode_id="trajectory-1",
                config=VERIFIED,
            )


if __name__ == "__main__":
    unittest.main()
