import unittest
from conditional_sequence_supervisor_v2 import ConditionalWindowMonitor
class Tests(unittest.TestCase):
 def test_sustained_robust_evidence_alarms(self):
  m=ConditionalWindowMonitor(window=3,alarm_threshold=2,reset_threshold=1)
  for _ in range(3):r=m.update(4,1,1,1)
  self.assertTrue(r['execution_alarm']);self.assertEqual(r['window_score'],3)
 def test_model_unknown_does_not_erase_physical_evidence(self):
  m=ConditionalWindowMonitor(window=3,alarm_threshold=2,reset_threshold=1)
  for _ in range(3):r=m.update(4,1,1,0)
  self.assertTrue(r['model_unknown']);self.assertTrue(r['execution_alarm']);self.assertFalse(r['sensor_protective_stop'])
 def test_sensor_fault_is_separate_and_not_scored(self):
  m=ConditionalWindowMonitor(window=3,alarm_threshold=2,reset_threshold=1,sensor_invalid_patience=2)
  r=m.update(4,1,0,1);r=m.update(4,1,0,1)
  self.assertTrue(r['sensor_protective_stop']);self.assertIsNone(r['window_score'])
 def test_isolated_event_diluted_and_hysteresis(self):
  m=ConditionalWindowMonitor(window=3,alarm_threshold=2,reset_threshold=1)
  for value in (0,5,0):r=m.update(value,0,1,1)
  self.assertFalse(r['execution_alarm'])
if __name__=='__main__':unittest.main()
