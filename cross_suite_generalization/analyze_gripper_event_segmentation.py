#!/usr/bin/env python3
"""Audit gripper close-event segmentation without outcome labels."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def hysteresis_events(commands: list[float], minimum_interval: int) -> list[int]:
    events: list[int] = []
    armed = True
    for index, command in enumerate(commands):
        if command <= -0.5:
            armed = True
        elif command >= 0.5 and armed:
            if not events or index - events[-1] >= minimum_interval:
                events.append(index)
            armed = False
    return events


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot-state", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = json.loads(args.robot_state.read_text(encoding="utf-8"))["records"]
    intervals = (0, 5, 10, 20, 40)
    output = []
    for row in records:
        commands = [float(value) for value in row["gripper_command"]]
        variants = {str(gap): hysteresis_events(commands, gap) for gap in intervals}
        base = variants["0"]
        output.append({
            "anonymous_id": row["anonymous_id"],
            "num_frames": len(commands),
            "events_by_minimum_interval": variants,
            "inter_event_gaps": [b - a for a, b in zip(base, base[1:])],
        })
    summary = {
        str(gap): {
            "total_events": sum(len(row["events_by_minimum_interval"][str(gap)]) for row in output),
            "episodes_with_multiple_events": sum(len(row["events_by_minimum_interval"][str(gap)]) > 1 for row in output),
        }
        for gap in intervals
    }
    result = {"schema_version": 1, "thresholds": {"open": -0.5, "close": 0.5}, "summary": summary, "records": output}
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
