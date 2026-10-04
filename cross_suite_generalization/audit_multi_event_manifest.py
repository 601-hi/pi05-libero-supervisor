"""Audit coverage and control availability without treating windows as independent."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    expected_controls = int(manifest["parameters"]["events_per_episode"])
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in manifest["episodes"]:
        groups[f'{row["suite"]}:task{row["task_id"]}'].append(row)

    group_rows = []
    for name, rows in sorted(groups.items()):
        events = [event for row in rows for event in row["label_blind_low_response_events"]]
        controls = [control for row in rows for control in row["label_blind_matched_controls"]]
        group_rows.append({
            "group": name,
            "episodes": len(rows),
            "failures": sum(not bool(row["success"]) for row in rows),
            "events": len(events),
            "controls": len(controls),
            "episodes_with_full_controls": sum(
                len(row["label_blind_matched_controls"]) == expected_controls for row in rows
            ),
            "event_response_median": float(np.median([
                event["response_ratio"] for event in events
            ])) if events else None,
            "control_response_median": float(np.median([
                control["response_ratio"] for control in controls
            ])) if controls else None,
        })

    result = {
        "manifest": str(args.manifest.resolve()),
        "episode_level": {
            "episodes": len(manifest["episodes"]),
            "failures": sum(not bool(row["success"]) for row in manifest["episodes"]),
            "control_count_distribution": dict(Counter(
                str(len(row["label_blind_matched_controls"])) for row in manifest["episodes"]
            )),
        },
        "warning": (
            "Event/control rows diagnose mechanisms but are clustered within episodes. "
            "Do not calculate evaluation confidence intervals from their raw count."
        ),
        "groups": group_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result["episode_level"], ensure_ascii=False))


if __name__ == "__main__":
    main()
