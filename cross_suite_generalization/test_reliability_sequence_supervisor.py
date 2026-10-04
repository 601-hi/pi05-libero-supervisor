import unittest
import json
import tempfile
from pathlib import Path
import numpy as np

from reliability_sequence_supervisor import (
    ReliabilityCalibration, SensorQualityMonitor, SequentialRiskAccumulator,
)


class ReliabilityTests(unittest.TestCase):
    def setUp(self):
        self.cal = ReliabilityCalibration(
            best_logp_reference=np.linspace(-20, -2, 200),
            ensemble_std_reference=np.linspace(0, 1, 200),
            normal_lr_threshold=-1,
            abnormal_lr_threshold=1,
            minimum_reliability=.1,
        )

    def test_four_decision_states(self):
        self.assertEqual(self.cal.score(-3, -8, .05, 1)["decision"], "known_normal")
        self.assertEqual(self.cal.score(-8, -3, .05, 1)["decision"], "known_abnormal")
        self.assertEqual(self.cal.score(-3, -3.2, .05, 1)["decision"], "ambiguous_overlap")
        self.assertEqual(self.cal.score(-100, -99, .05, 1)["decision"], "unknown")

    def test_disagreement_and_bad_sensor_abstain(self):
        self.assertEqual(self.cal.score(-3, -8, 2, 1)["decision"], "unknown")
        self.assertEqual(self.cal.score(-3, -8, .05, 0)["decision"], "unknown")

    def test_nonfinite_is_unknown(self):
        self.assertEqual(self.cal.score(float("nan"), -3, .1, 1)["decision"], "unknown")

    def test_loader_rejects_target_or_overlapping_calibration(self):
        base = {"source_only": False, "overlap_warning": False,
                "best_logp_reference": [-2, -1], "ensemble_std_reference": [0, 1],
                "normal_lr_threshold": -1, "abnormal_lr_threshold": 1}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cal.json"
            path.write_text(json.dumps(base), encoding="utf-8")
            with self.assertRaises(ValueError): ReliabilityCalibration.from_json(path)
            base["source_only"] = True; base["overlap_warning"] = True
            path.write_text(json.dumps(base), encoding="utf-8")
            with self.assertRaises(ValueError): ReliabilityCalibration.from_json(path)

    def test_sensor_quality(self):
        monitor = SensorQualityMonitor(expected_dt=.05)
        self.assertEqual(monitor.score(.05, .01, 1), 1)
        self.assertEqual(monitor.score(.10, .01, 1), 0)
        self.assertEqual(monitor.score(.05, .20, 1), 0)


class SequenceTests(unittest.TestCase):
    def test_isolated_spike_does_not_alarm(self):
        acc = SequentialRiskAccumulator(alarm_threshold=3, reset_threshold=.2, drift=.3, decay=.8)
        for lr in [0, 0, 2, -2, -2]:
            result = acc.update(lr, 1, "known_abnormal" if lr > 1 else "known_normal")
        self.assertFalse(result["alarm"])

    def test_sustained_evidence_alarms(self):
        acc = SequentialRiskAccumulator(alarm_threshold=3, reset_threshold=.2, drift=.2, decay=.95)
        outputs = [acc.update(1.5, .9, "known_abnormal") for _ in range(4)]
        self.assertTrue(outputs[-1]["alarm"])

    def test_unknown_is_not_treated_as_normal(self):
        acc = SequentialRiskAccumulator(alarm_threshold=3, reset_threshold=.2, drift=.2, unknown_patience=3)
        outputs = [acc.update(float("nan"), 0, "unknown") for _ in range(3)]
        self.assertTrue(outputs[-1]["protective_observation"])
        self.assertEqual(outputs[-1]["unknown_run"], 3)

    def test_hysteresis(self):
        acc = SequentialRiskAccumulator(alarm_threshold=2, reset_threshold=.5, drift=0, decay=1)
        acc.update(3, 1, "known_abnormal")
        self.assertTrue(acc.alarm)
        acc.update(-1, 1, "known_normal")
        self.assertTrue(acc.alarm)
        acc.update(-3, 1, "known_normal")
        self.assertFalse(acc.alarm)


if __name__ == "__main__":
    unittest.main()
