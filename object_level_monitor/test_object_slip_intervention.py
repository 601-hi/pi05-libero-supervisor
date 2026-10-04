import unittest

import numpy as np

from object_slip_intervention import ObjectSlipIntervention, OracleLiftTrigger


class FakeModel:
    def __init__(self):
        self.geom_friction = np.asarray([[0.95, 0.3, 0.1], [0.7, 0.2, 0.05]])
        self.body_mass = np.asarray([0.0, 2.0])

    def body_name2id(self, name):
        return {"target_root": 1}[name]

    def geom_name2id(self, name):
        return {"target_g0": 0, "target_g1": 1}[name]


class FakeData:
    def __init__(self):
        self.xfrc_applied = np.zeros((2, 6))


class FakeSim:
    def __init__(self):
        self.model = FakeModel()
        self.data = FakeData()


class ObjectSlipInterventionTest(unittest.TestCase):
    def test_apply_and_restore_only_target_parameters(self):
        sim = FakeSim()
        original = sim.model.geom_friction.copy()
        intervention = ObjectSlipIntervention(
            sim,
            root_body_name="target_root",
            contact_geom_names=["target_g0", "target_g1"],
            friction_scale=0.1,
            downward_force_weight_ratio=0.5,
        )
        result = intervention.apply(True)
        np.testing.assert_allclose(sim.model.geom_friction, original * 0.1)
        self.assertAlmostEqual(sim.data.xfrc_applied[1, 2], -9.81)
        self.assertEqual(result.applied_force_world_n, (0.0, 0.0, -9.81))
        intervention.restore()
        np.testing.assert_allclose(sim.model.geom_friction, original)
        np.testing.assert_allclose(sim.data.xfrc_applied, 0.0)

    def test_oracle_lift_trigger_is_persistent_and_causal(self):
        trigger = OracleLiftTrigger(0.1, rise_threshold_m=0.03, consecutive=2)
        self.assertFalse(trigger.update(0.14, False))
        self.assertFalse(trigger.update(0.129, True))
        self.assertFalse(trigger.update(0.131, True))
        self.assertTrue(trigger.update(0.132, True))
        self.assertTrue(trigger.update(0.10, False))

    def test_invalid_intervention_parameters_are_rejected(self):
        with self.assertRaises(ValueError):
            ObjectSlipIntervention(
                FakeSim(),
                root_body_name="target_root",
                contact_geom_names=["target_g0"],
                friction_scale=1.1,
            )


if __name__ == "__main__":
    unittest.main()
