"""Calibrate and evaluate a causal finite-time gripper close-response gate."""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np


def load_episodes(paths: list[Path]) -> list[tuple[str, list[dict]]]:
    result = []
    for source_id, path in enumerate(paths):
        groups: dict[tuple[int, int], list[dict]] = collections.defaultdict(list)
        with path.open("r", encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                if row.get("event") == "step":
                    groups[(int(row["task_id"]), int(row["episode_idx"]))].append(row)
        for (task, episode), rows in sorted(groups.items()):
            rows.sort(key=lambda row: int(row["action_index"]))
            result.append((f"source{source_id}:task{task}:episode{episode}", rows))
    return result


def width(row: dict, when: str) -> float:
    qpos = np.asarray(row[f"gripper_qpos_{when}"], dtype=np.float64)
    return float(qpos[0] - qpos[1])


def close_events(
    episodes: list[tuple[str, list[dict]]],
    open_width_min: float,
    close_threshold: float,
    consecutive: int,
    horizon: int,
) -> list[dict]:
    events = []
    for episode_id, rows in episodes:
        run = 0
        armed = True
        for i, row in enumerate(rows):
            close = float(row["intended_action"][-1]) >= close_threshold
            run = run + 1 if close else 0
            if not close:
                armed = True
            if run != consecutive or not armed:
                continue
            start = i - consecutive + 1
            armed = False
            if start + horizon > len(rows):
                continue
            width_before = width(rows[start], "before")
            if width_before < open_width_min:
                continue
            end = start + horizon - 1
            closure = width_before - width(rows[end], "after")
            active = any(bool(r.get("gripper_disturbance_applied", False)) for r in rows[start : end + 1])
            events.append(
                {
                    "episode_id": episode_id,
                    "start_action": int(rows[start]["action_index"]),
                    "decision_action": int(rows[end]["action_index"]),
                    "width_before": width_before,
                    "closure": closure,
                    "overlaps_applied_fault": active,
                }
            )
    return events


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration-trace", action="append", type=Path, required=True)
    parser.add_argument("--evaluation-trace", action="append", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--normal-event-budget", type=int, default=1)
    parser.add_argument("--open-width-min", type=float, default=0.04)
    parser.add_argument("--close-threshold", type=float, default=0.5)
    parser.add_argument("--consecutive", type=int, default=2)
    parser.add_argument("--horizon", type=int, default=3)
    args = parser.parse_args()

    calibration = close_events(
        load_episodes(args.calibration_trace),
        args.open_width_min,
        args.close_threshold,
        args.consecutive,
        args.horizon,
    )
    if any(event["overlaps_applied_fault"] for event in calibration):
        raise ValueError("calibration contains applied gripper faults")
    closure = np.sort(np.asarray([event["closure"] for event in calibration]))
    if len(closure) <= args.normal_event_budget:
        raise ValueError("not enough calibration events for requested budget")
    # Alarm when closure is strictly below the selected order statistic.
    threshold = float(closure[args.normal_event_budget])
    evaluation = close_events(
        load_episodes(args.evaluation_trace),
        args.open_width_min,
        args.close_threshold,
        args.consecutive,
        args.horizon,
    )
    for event in evaluation:
        event["alarm"] = bool(event["closure"] < threshold)
    normal = [event for event in evaluation if not event["overlaps_applied_fault"]]
    fault = [event for event in evaluation if event["overlaps_applied_fault"]]
    report = {
        "version": 1,
        "causal_decision_delay_steps": args.horizon,
        "open_width_min": args.open_width_min,
        "close_threshold": args.close_threshold,
        "consecutive_close_commands": args.consecutive,
        "minimum_closure_threshold": threshold,
        "normal_event_budget": args.normal_event_budget,
        "calibration_events": len(calibration),
        "calibration_false_alarms": int(sum(event["closure"] < threshold for event in calibration)),
        "evaluation_events": len(evaluation),
        "normal_events": len(normal),
        "normal_event_false_alarms": int(sum(event["alarm"] for event in normal)),
        "fault_events": len(fault),
        "fault_event_detections": int(sum(event["alarm"] for event in fault)),
        "events": evaluation,
    }
    text = json.dumps(report, ensure_ascii=False, indent=2)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
