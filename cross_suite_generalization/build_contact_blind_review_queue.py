#!/usr/bin/env python3
"""Build a strictly blinded review queue from fixed-view motion evidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion-evidence", type=Path, required=True)
    parser.add_argument("--sheet-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compromised-id", action="append", default=[])
    args = parser.parse_args()
    evidence = json.loads(args.motion_evidence.read_text(encoding="utf-8"))["records"]
    compromised = set(args.compromised_id)
    rows = []
    for episode in evidence:
        identifier = episode["anonymous_id"]
        for event_index, event in enumerate(episode["events"]):
            close = int(event["close_frame"])
            groups = event["co_motion_groups"]
            sustained = [candidate for candidate in event["candidates"] if candidate["sustained_motion"]]
            visibility = max((candidate["postclose_visible_fraction"] for candidate in sustained), default=0.0)
            category = "single_motion_mode" if len(groups) == 1 else "no_motion_mode" if not groups else "multiple_motion_modes"
            sheet = args.sheet_dir / f"{identifier}_close{event_index:02d}_f{close:04d}_motion.jpg"
            if not sheet.exists():
                raise FileNotFoundError(sheet)
            rows.append({
                "anonymous_id": identifier,
                "event_index": event_index,
                "close_frame": close,
                "review_category": category,
                "co_motion_group_count": len(groups),
                "largest_group_size": max(map(len, groups), default=0),
                "best_track_visibility": visibility,
                "sheet": sheet.name,
                "blindness_status": "compromised_exclude_from_unbiased_pass_a" if identifier in compromised else "eligible",
                "annotation": None,
            })
    priority = {"single_motion_mode": 0, "no_motion_mode": 1, "multiple_motion_modes": 2}
    rows.sort(key=lambda row: (
        row["blindness_status"] != "eligible",
        priority[row["review_category"]],
        -row["best_track_visibility"],
        row["anonymous_id"],
        row["event_index"],
    ))
    output = {
        "schema_version": 1,
        "annotation_pass": "A_physical_blind",
        "forbidden_information": ["task language", "episode outcome", "reward", "semantic class", "expert score"],
        "warning": "Motion category is review prioritization only and must not be copied as ground truth.",
        "records": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    counts = {}
    for row in rows:
        counts[row["review_category"]] = counts.get(row["review_category"], 0) + 1
    print(json.dumps({"events": len(rows), "categories": counts, "compromised": sum(row["blindness_status"] != "eligible" for row in rows)}, indent=2))


if __name__ == "__main__":
    main()
