import unittest

import numpy as np

from common_dynamics_transition import (
    CausalTransition,
    CommandSemantics,
    require_valid_transition,
    validate_transition,
)


def valid_row(**updates):
    values = dict(
        dataset="external_example",
        episode_id="episode-1",
        step_id=0,
        robot_model="panda",
        controller_mode="cartesian_velocity",
        command_source="recorded_action_dict.cartesian_velocity",
        timing_source="measured_timestamps",
        command_timestamp=1.01,
        state_before_timestamp=1.0,
        state_after_timestamp=1.02,
        command=np.zeros(7),
        command_semantics=CommandSemantics(
            mode="eef_velocity",
            reference_frame="robot_base",
            units="m/s,rad/s,normalized_gripper",
            rotation_representation="axis_angle_velocity",
            gripper_semantics="absolute_open_fraction",
        ),
        state_before={"eef_pose": np.zeros(7), "joint_position": np.zeros(7)},
        state_after={"eef_pose": np.ones(7), "joint_position": np.ones(7)},
        quality_flags={"timestamp_monotonic": True, "units_verified": True},
    )
    values.update(updates)
    return CausalTransition(**values)


class CommonDynamicsTransitionTest(unittest.TestCase):
    def test_valid_transition_passes(self):
        row = valid_row()
        self.assertIs(require_valid_transition(row), row)

    def test_command_later_than_response_is_rejected(self):
        errors = validate_transition(valid_row(command_timestamp=1.03))
        self.assertTrue(any("later than" in error for error in errors))

    def test_historical_command_before_response_interval_is_allowed(self):
        self.assertEqual(validate_transition(valid_row(command_timestamp=0.90)), [])

    def test_oracle_and_success_features_are_rejected(self):
        errors = validate_transition(
            valid_row(optional_sensors={"oracle_only_object_pose": [0, 0, 0], "success": True})
        )
        self.assertEqual(sum("forbidden deployable" in error for error in errors), 2)

    def test_ambiguous_command_mode_is_rejected(self):
        semantics = valid_row().command_semantics
        errors = validate_transition(
            valid_row(command_semantics=CommandSemantics(
                mode="unknown_7d",
                reference_frame=semantics.reference_frame,
                units=semantics.units,
                rotation_representation=semantics.rotation_representation,
                gripper_semantics=semantics.gripper_semantics,
            ))
        )
        self.assertTrue(any("unsupported command mode" in error for error in errors))

    def test_failed_quality_flag_is_rejected(self):
        errors = validate_transition(valid_row(quality_flags={"units_verified": False}))
        self.assertTrue(any("units_verified" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
