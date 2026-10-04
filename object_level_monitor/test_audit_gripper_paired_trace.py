import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


def row(i, grip, *, applied=False, oracle=True, kine=True):
    result = {
        "event": "step",
        "action_index": i,
        "intended_action": [0.1, 0, 0, 0, 0, 0, 1.0 if i >= 2 else -1.0],
        "executed_action": [0.1, 0, 0, 0, 0, 0, grip],
        "eef_pos_before": [0.01 * i, 0, 0],
        "eef_pos_after": [0.01 * (i + 1), 0, 0],
        "gripper_disturbance_applied": applied,
    }
    if oracle:
        result["oracle_only_object_state_before"] = [0, 0, 0]
        result["oracle_only_object_state_after"] = [0, 0, 0]
    if kine:
        result["kinematic_safety_before"] = {"sigma_min": 0.2}
    return result


class PairedAuditTest(unittest.TestCase):
    def test_valid_pair(self):
        with tempfile.TemporaryDirectory() as tmp:
            normal = Path(tmp) / "normal.jsonl"
            abnormal = Path(tmp) / "abnormal.jsonl"
            nrows = [row(i, 1.0 if i >= 2 else -1.0) for i in range(4)]
            arows = [row(i, -1.0 if i >= 2 else -1.0, applied=i >= 2) for i in range(4)]
            normal.write_text("".join(json.dumps(x) + "\n" for x in nrows), encoding="utf-8")
            abnormal.write_text("".join(json.dumps(x) + "\n" for x in arows), encoding="utf-8")
            result = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).with_name("audit_gripper_paired_trace.py")),
                    "--normal",
                    str(normal),
                    "--intervention",
                    str(abnormal),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            report = json.loads(result.stdout)
            self.assertTrue(report["causal_pre_match"])
            self.assertTrue(report["isolated_at_onset"])


if __name__ == "__main__":
    unittest.main()
