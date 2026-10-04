import unittest

import numpy as np

from droid_rlds_adapter import DroidRldsConfig, adapt_episode


VERIFIED_CONFIG = DroidRldsConfig(
    command_semantics_verified=True,
    units_verified=True,
    response_lag_steps=0,
    timing_alignment_verified=True,
)


def step(offset=0.0):
    return {
        "observation": {
            "cartesian_position": np.arange(6, dtype=float) + offset,
            "joint_position": np.arange(7, dtype=float) + offset,
            "gripper_position": np.asarray([0.8 - offset]),
        },
        "action_dict": {
            "cartesian_velocity": np.full(6, 0.1 + offset),
            "gripper_position": np.asarray([0.7]),
        },
    }


class DroidRldsAdapterTest(unittest.TestCase):
    def test_adjacent_steps_form_one_causal_transition(self):
        rows = adapt_episode([step(0.0), step(0.01)], episode_id="e1", config=VERIFIED_CONFIG)
        self.assertEqual(len(rows), 1)
        np.testing.assert_allclose(rows[0].state_before["joint_position"], np.arange(7))
        np.testing.assert_allclose(rows[0].state_after["joint_position"], np.arange(7) + 0.01)
        self.assertAlmostEqual(rows[0].delta_t_seconds, 1 / 15)
        self.assertIn("fixed_cadence_assumption", rows[0].timing_source)

    def test_final_step_is_not_emitted_without_after_state(self):
        self.assertEqual(adapt_episode([step()], episode_id="e1", config=VERIFIED_CONFIG), [])

    def test_missing_command_field_is_rejected(self):
        first = step()
        del first["action_dict"]["cartesian_velocity"]
        with self.assertRaises(KeyError):
            adapt_episode([first, step(0.01)], episode_id="e1", config=VERIFIED_CONFIG)

    def test_bad_frequency_is_rejected(self):
        with self.assertRaises(ValueError):
            adapt_episode([step(), step(0.01)], episode_id="e1", config=DroidRldsConfig(control_frequency_hz=0))

    def test_command_is_velocity_plus_absolute_gripper_position(self):
        row = adapt_episode([step(), step(0.01)], episode_id="e1", config=VERIFIED_CONFIG)[0]
        np.testing.assert_allclose(row.command[:6], 0.1)
        self.assertAlmostEqual(row.command[-1], 0.7)
        self.assertEqual(row.command_semantics.mode, "eef_velocity")

    def test_unverified_default_config_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "response_lag_steps|timing_alignment_verified"):
            adapt_episode([step(), step(0.01)], episode_id="e1")

    def test_positive_response_lag_preserves_causality(self):
        config = DroidRldsConfig(
            command_semantics_verified=True,
            units_verified=True,
            response_lag_steps=2,
            timing_alignment_verified=True,
        )
        rows = adapt_episode(
            [step(0.00), step(0.01), step(0.02), step(0.03)], episode_id="e1", config=config
        )
        self.assertEqual(len(rows), 1)
        np.testing.assert_allclose(rows[0].command[:6], 0.1)
        np.testing.assert_allclose(rows[0].state_before["joint_position"], np.arange(7) + 0.02)
        self.assertLess(rows[0].command_timestamp, rows[0].state_before_timestamp)


if __name__ == "__main__":
    unittest.main()
