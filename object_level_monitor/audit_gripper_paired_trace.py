"""Audit causal alignment and intervention isolation for a paired rollout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_steps(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as stream:
        return [row for line in stream if (row := json.loads(line)).get("event") == "step"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--normal", type=Path, required=True)
    parser.add_argument("--intervention", type=Path, required=True)
    args = parser.parse_args()

    normal = load_steps(args.normal)
    abnormal = load_steps(args.intervention)
    active = [r for r in abnormal if r.get("gripper_disturbance_applied")]
    if not active:
        raise SystemExit("No applied gripper intervention was recorded")
    onset = int(active[0]["action_index"])
    n_by_i = {int(r["action_index"]): r for r in normal}
    a_by_i = {int(r["action_index"]): r for r in abnormal}
    common_pre = sorted(set(n_by_i) & set(a_by_i) & set(range(onset)))
    if not common_pre:
        raise SystemExit("No common pre-intervention steps")

    def max_diff(field: str, indices: list[int]) -> float:
        return max(
            float(np.max(np.abs(np.asarray(n_by_i[i][field]) - np.asarray(a_by_i[i][field]))))
            for i in indices
        )

    report = {
        "onset": onset,
        "common_pre_steps": len(common_pre),
        "pre_intended_action_max_diff": max_diff("intended_action", common_pre),
        "pre_executed_action_max_diff": max_diff("executed_action", common_pre),
        "pre_eef_before_max_diff": max_diff("eef_pos_before", common_pre),
        "pre_eef_after_max_diff": max_diff("eef_pos_after", common_pre),
        "onset_intended_action_max_diff": max_diff("intended_action", [onset]),
        "onset_non_gripper_executed_max_diff": float(
            np.max(
                np.abs(
                    np.asarray(n_by_i[onset]["executed_action"][:-1])
                    - np.asarray(a_by_i[onset]["executed_action"][:-1])
                )
            )
        ),
        "onset_gripper_executed_difference": float(
            abs(
                float(n_by_i[onset]["executed_action"][-1])
                - float(a_by_i[onset]["executed_action"][-1])
            )
        ),
        "applied_steps": len(active),
        "oracle_fields_present": all(
            "oracle_only_object_state_before" in row and "oracle_only_object_state_after" in row
            for row in abnormal
        ),
        "kinematic_fields_present": all("kinematic_safety_before" in row for row in abnormal),
    }
    report["causal_pre_match"] = all(
        report[key] == 0.0
        for key in (
            "pre_intended_action_max_diff",
            "pre_executed_action_max_diff",
            "pre_eef_before_max_diff",
            "pre_eef_after_max_diff",
        )
    )
    report["isolated_at_onset"] = (
        report["onset_intended_action_max_diff"] == 0.0
        and report["onset_non_gripper_executed_max_diff"] == 0.0
        and report["onset_gripper_executed_difference"] > 0.0
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["causal_pre_match"] or not report["isolated_at_onset"]:
        raise SystemExit("Paired trace failed causal/isolation audit")


if __name__ == "__main__":
    main()
