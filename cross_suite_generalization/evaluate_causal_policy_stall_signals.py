#!/usr/bin/env python3
"""Evaluate causal policy-stall signals on natural runs without outcome leakage.

Calibration uses successful episodes outside the held-out trace files.  The held-out
success/failure outcomes are read only after thresholds have been fixed.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict, deque
from pathlib import Path

import numpy as np


def conformal_upper(values, alpha):
    values = np.sort(np.asarray(values, float))
    rank = min(len(values), math.ceil((len(values) + 1) * (1 - alpha)))
    return float(values[rank - 1]), rank


def suite_from_path(path: Path) -> str:
    text = str(path).lower()
    for name in ("libero_90", "libero_10", "libero_goal", "libero_object", "libero_spatial"):
        if name in text:
            return name
    return "unknown"


def load_episodes(root: Path):
    episodes = []
    for path in sorted(root.rglob("*.jsonl")):
        steps, endings = defaultdict(list), {}
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if row.get("event") == "step":
                    steps[(int(row["task_id"]), int(row["episode_idx"]))].append(row)
                elif row.get("event") == "episode_end":
                    endings[(int(row["task_id"]), int(row["episode_idx"]))] = row
        for key, rows in steps.items():
            if key not in endings or not rows:
                continue
            rows.sort(key=lambda row: int(row["action_index"]))
            if any(bool(row.get("disturbance_active", False)) for row in rows):
                continue
            episodes.append({"path": path, "suite": suite_from_path(path), "task_id": key[0],
                             "episode_idx": key[1], "success": bool(endings[key].get("success", False)),
                             "rows": rows})
    return episodes


def causal_signals(rows, reversal_window=40):
    run = 0
    reversal_history = deque()
    cumulative_reversals = 0
    previous_gripper = None
    output = {"low_progress_run": [], "gripper_reversals_window": [],
              "gripper_reversals_cumulative": [], "elapsed_steps": []}
    for i, row in enumerate(rows):
        command = np.asarray(row.get("intended_target_translation", np.zeros(3)), float)
        actual = np.asarray(row.get("actual_translation", np.zeros(3)), float)
        norm2 = float(command @ command)
        progress = float(command @ actual / (norm2 + 1e-12))
        low = norm2 > 0.015 ** 2 and progress < 0.05
        run = run + 1 if low else 0
        grip = float(row.get("intended_action", row.get("action", np.zeros(7)))[6])
        reversal = (previous_gripper is not None and grip * previous_gripper < 0 and
                    abs(grip) > 0.5 and abs(previous_gripper) > 0.5)
        reversal_history.append(int(reversal))
        cumulative_reversals += int(reversal)
        if len(reversal_history) > reversal_window:
            reversal_history.popleft()
        previous_gripper = grip
        output["low_progress_run"].append(run)
        output["gripper_reversals_window"].append(sum(reversal_history))
        output["gripper_reversals_cumulative"].append(cumulative_reversals)
        output["elapsed_steps"].append(i + 1)
    return {key: np.asarray(value, float) for key, value in output.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-root", type=Path, required=True)
    parser.add_argument("--holdout-substring", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=0.10)
    args = parser.parse_args()
    episodes = load_episodes(args.trace_root)
    def held_out(ep):
        return any(token in str(ep["path"]) for token in args.holdout_substring)
    calibration = [ep for ep in episodes if ep["success"] and not held_out(ep)]
    target = [ep for ep in episodes if held_out(ep)]
    signal_names = tuple(causal_signals(target[0]["rows"]).keys())
    results = []
    for name in signal_names:
        maxima = [float(causal_signals(ep["rows"])[name].max()) for ep in calibration]
        threshold, rank = conformal_upper(maxima, args.alpha)
        details = []
        for ep in target:
            values = causal_signals(ep["rows"])[name]
            alarm = values > threshold
            hit = np.flatnonzero(alarm)
            details.append({"path": str(ep["path"]), "suite": ep["suite"], "task_id": ep["task_id"],
                            "episode_idx": ep["episode_idx"], "success": ep["success"],
                            "steps": len(values), "maximum": float(values.max()),
                            "alarm": bool(len(hit)), "first_alarm_action_index": int(hit[0]) if len(hit) else None})
        success = [x for x in details if x["success"]]
        failure = [x for x in details if not x["success"]]
        results.append({"signal": name, "threshold": threshold, "calibration_rank": rank,
                        "success_alarm_rate": float(np.mean([x["alarm"] for x in success])) if success else None,
                        "failure_detection_rate": float(np.mean([x["alarm"] for x in failure])) if failure else None,
                        "details": details})
    payload = {"evaluation_type": "cross-task frozen causal diagnostic",
               "target_used_for_threshold_selection": False, "alpha": args.alpha,
               "calibration_success_episodes": len(calibration), "target_episodes": len(target),
               "holdout_substrings": args.holdout_substring, "results": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({**{k: payload[k] for k in ("calibration_success_episodes", "target_episodes")},
                      "results": [{k: v for k, v in row.items() if k != "details"} for row in results]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
