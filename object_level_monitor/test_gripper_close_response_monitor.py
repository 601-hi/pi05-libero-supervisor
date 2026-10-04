import unittest

from gripper_close_response_monitor import GripperCloseResponseMonitor, MonitorState


class GripperCloseResponseMonitorTest(unittest.TestCase):
    def test_normal_close_does_not_alarm(self) -> None:
        monitor = GripperCloseResponseMonitor(minimum_closure=0.002)
        results = [
            monitor.update(1.0, 0.080, 0.079),
            monitor.update(1.0, 0.079, 0.077),
            monitor.update(1.0, 0.077, 0.074),
        ]
        self.assertFalse(any(result.alarm for result in results))
        self.assertIs(results[-1].state, MonitorState.IDLE)

    def test_failed_close_latches_single_alarm(self) -> None:
        monitor = GripperCloseResponseMonitor(minimum_closure=0.002)
        results = [
            monitor.update(1.0, 0.08000, 0.08001),
            monitor.update(1.0, 0.08001, 0.08002),
            monitor.update(1.0, 0.08002, 0.08003),
        ]
        self.assertTrue(results[-1].new_alarm)
        self.assertIs(results[-1].state, MonitorState.LATCHED)
        repeated = [monitor.update(-1.0, 0.08003, 0.08002) for _ in range(10)]
        self.assertTrue(all(result.alarm for result in repeated))
        self.assertFalse(any(result.new_alarm for result in repeated))

    def test_normal_event_rearms_only_after_open_persistence(self) -> None:
        monitor = GripperCloseResponseMonitor(minimum_closure=0.002)
        monitor.update(1.0, 0.080, 0.079)
        monitor.update(1.0, 0.079, 0.077)
        result = monitor.update(1.0, 0.077, 0.074)
        self.assertIs(result.state, MonitorState.IDLE)
        self.assertIs(monitor.update(-1.0, 0.074, 0.075).state, MonitorState.IDLE)
        self.assertIs(monitor.update(-1.0, 0.075, 0.078).state, MonitorState.ARMED)

    def test_explicit_reset_clears_latch(self) -> None:
        monitor = GripperCloseResponseMonitor(minimum_closure=0.002)
        monitor.update(1.0, 0.080, 0.080)
        monitor.update(1.0, 0.080, 0.080)
        self.assertTrue(monitor.update(1.0, 0.080, 0.080).alarm)
        monitor.reset()
        self.assertIs(monitor.state, MonitorState.ARMED)


if __name__ == "__main__":
    unittest.main()
