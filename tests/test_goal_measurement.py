import unittest
from dataclasses import replace
from vla_supervisor.goal_measurement import ValidatedImagePoint, approach_distance


class TestGoalMeasurement(unittest.TestCase):
    def setUp(self):
        self.g = ValidatedImagePoint((.1,.2), .01, 40, 'fixed-v1', 'fingertip_midpoint', '', 'review-v1')
        self.t = ValidatedImagePoint((.4,.6), .02, 40, 'fixed-v1', 'grasp_reference', 'bowl-a', 'review-v1')

    def run_distance(self, g=None, t=None):
        return approach_distance(g or self.g, t or self.t,
            trusted_validation_ids={'review-v1'}, fixed_domain='fixed-v1')

    def test_geometry(self):
        r = self.run_distance()
        self.assertAlmostEqual(r.value, .5)
        self.assertAlmostEqual(r.error_bound, .03)

    def test_untrusted_model_claim(self):
        self.assertEqual(self.run_distance(t=replace(self.t, validation_id='vlm-confident')).reason, 'unvalidated')

    def test_wrong_point(self):
        self.assertEqual(self.run_distance(g=replace(self.g, semantic_role='arm_box_center')).reason, 'not_fingertip_midpoint')

    def test_domains(self):
        self.assertEqual(self.run_distance(t=replace(self.t, domain='wrist')).reason, 'domain_mismatch')

    def test_alignment(self):
        self.assertEqual(self.run_distance(t=replace(self.t, action_index=41)).reason, 'time_mismatch')

    def test_occlusion(self):
        self.assertEqual(self.run_distance(t=replace(self.t, visible=False)).reason, 'occluded')

    def test_uncertainty(self):
        for radius in (-1, float('nan'), float('inf')):
            self.assertIsNone(self.run_distance(t=replace(self.t, error_radius=radius)).value)

    def test_bad_coords(self):
        for xy in ((-1,.1), (float('nan'),0), (0,2)):
            self.assertIsNone(self.run_distance(t=replace(self.t, xy=xy)).value)

    def test_box_center_not_grasp_reference(self):
        self.assertIsNone(self.run_distance(t=replace(self.t, semantic_role='object_box_center')).value)

    def test_missing(self):
        self.assertIsNone(approach_distance(None, self.t, trusted_validation_ids=set(), fixed_domain='fixed-v1').value)


if __name__ == '__main__':
    unittest.main()
