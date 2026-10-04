#!/usr/bin/env python3
"""Audit episode outcomes and motion-quality proxies in existing JSONL traces."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def longest_true_run(values):
    best = current = 0
    for value in values:
        current = current + 1 if value else 0
        best = max(best, current)
    return best


def episode_metrics(path, rows, end, control_hz):
    rows.sort(key=lambda x: int(x["action_index"]))
    actual = np.asarray([r.get("actual_translation", np.zeros(3)) for r in rows], float)
    q_before = np.asarray([r["joint_pos_before"] for r in rows], float)
    q_after = np.asarray([r["joint_pos_after"] for r in rows], float)
    qvel = np.asarray([r["joint_vel_before"] for r in rows], float)
    gripvel = np.asarray([r["gripper_qvel_before"] for r in rows], float)
    gripper_command = np.asarray([r.get("intended_action", r.get("action", np.zeros(7)))[6] for r in rows], float)
    command = np.asarray([r.get("intended_target_translation", np.zeros(3)) for r in rows], float)
    cmd_norm = np.linalg.norm(command, axis=1)
    progress = np.sum(command * actual, axis=1) / (cmd_norm ** 2 + 1e-12)
    dt = 1.0 / control_hz
    qacc = np.diff(qvel, axis=0) / dt
    qjerk = np.diff(qacc, axis=0) / dt
    sign = np.sign(qvel)
    reversals = ((sign[1:] * sign[:-1] < 0) &
                 (np.abs(qvel[1:]) > 0.05) & (np.abs(qvel[:-1]) > 0.05)).sum()
    grip_sign = np.sign(gripper_command)
    gripper_command_reversals = int(((grip_sign[1:] * grip_sign[:-1] < 0) &
                                     (np.abs(gripper_command[1:]) > 0.5) &
                                     (np.abs(gripper_command[:-1]) > 0.5)).sum())
    abs_qjerk = np.abs(qjerk)
    active_steps = sum(bool(r.get("disturbance_active", False)) for r in rows)
    return {
        "path": str(path), "task_id": int(rows[0]["task_id"]),
        "episode_idx": int(rows[0]["episode_idx"]),
        "success": bool(end.get("success", False)), "steps": len(rows),
        "intervention_active_steps": int(active_steps),
        "natural": active_steps == 0,
        "eef_path_m": float(np.linalg.norm(actual, axis=1).sum()),
        "joint_path_rad": float(np.linalg.norm(q_after - q_before, axis=1).sum()),
        "max_abs_joint_velocity": float(np.abs(qvel).max()),
        "max_joint_velocity_norm": float(np.linalg.norm(qvel, axis=1).max()),
        "max_abs_joint_acceleration": float(np.abs(qacc).max()) if len(qacc) else 0.0,
        "max_abs_joint_jerk": float(np.abs(qjerk).max()) if len(qjerk) else 0.0,
        "p95_abs_joint_jerk": float(np.percentile(abs_qjerk, 95)) if len(qjerk) else 0.0,
        "rms_joint_jerk": float(np.sqrt(np.mean(qjerk ** 2))) if len(qjerk) else 0.0,
        "high_jerk_joint_events_gt100": int((abs_qjerk > 100.0).sum()) if len(qjerk) else 0,
        "max_gripper_velocity_norm": float(np.linalg.norm(gripvel, axis=1).max()),
        "gripper_command_reversals": gripper_command_reversals,
        "joint_velocity_reversals": int(reversals),
        "max_low_progress_run": int(longest_true_run((cmd_norm > 0.015) & (progress < 0.05))),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trace-directory", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--control-hz", type=float, default=20.0)
    args = p.parse_args()
    episodes = []
    errors = []
    for path in sorted(args.trace_directory.rglob("*.jsonl")):
        grouped = defaultdict(list); ends = {}
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    row = json.loads(line)
                    if row.get("event") == "step" and all(k in row for k in (
                        "task_id", "episode_idx", "action_index", "joint_pos_before",
                        "joint_pos_after", "joint_vel_before", "gripper_qvel_before")):
                        grouped[(int(row["task_id"]), int(row["episode_idx"]))].append(row)
                    elif row.get("event") == "episode_end":
                        ends[(int(row["task_id"]), int(row["episode_idx"]))] = row
            for key, rows in grouped.items():
                if key in ends and rows:
                    episodes.append(episode_metrics(path, rows, ends[key], args.control_hz))
        except Exception as exc:
            errors.append({"path": str(path), "error": repr(exc)})
    natural = [x for x in episodes if x["natural"]]
    disturbed = [x for x in episodes if not x["natural"]]
    metrics = ["steps", "eef_path_m", "joint_path_rad", "max_abs_joint_velocity",
               "max_abs_joint_acceleration", "max_abs_joint_jerk",
               "p95_abs_joint_jerk", "rms_joint_jerk", "high_jerk_joint_events_gt100",
               "gripper_command_reversals", "joint_velocity_reversals", "max_low_progress_run"]
    summary = {
        "files_scanned": len(list(args.trace_directory.rglob("*.jsonl"))),
        "episodes": len(episodes), "parse_errors": len(errors),
        "natural_episodes": len(natural),
        "natural_failures": sum(not x["success"] for x in natural),
        "disturbed_episodes": len(disturbed),
        "disturbed_failures": sum(not x["success"] for x in disturbed),
        "natural_metric_percentiles": {
            key: dict(zip(("p50", "p90", "p95", "p99"), map(float, np.percentile(
                [x[key] for x in natural], [50, 90, 95, 99]))))
            for key in metrics if natural
        },
    }
    result = {"summary": summary, "episodes": episodes, "errors": errors}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("NATURAL_FAILURE_EXAMPLES")
    for row in [x for x in natural if not x["success"]][:30]:
        print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
