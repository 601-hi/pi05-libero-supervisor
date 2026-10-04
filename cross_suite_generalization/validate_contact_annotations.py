"""Validate Pass-A physical contact annotations before any model fitting."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


ENTITY_TYPES = {"movable_object", "fixed_scene", "robot_self", "unknown"}
MECHANISMS = {
    "successful_pickup", "empty_grasp", "push_without_grasp", "fixed_obstacle_or_jam",
    "object_slip_or_loss", "planned_release", "unknown",
}
VISIBILITY = {"visible", "occluded", "uncertain"}
QUALITY = {"high", "medium", "low"}


def validate(payload: dict, index: dict) -> list[str]:
    errors = []
    if payload.get("annotation_pass") != "A_physical_blind":
        errors.append("annotation_pass must be A_physical_blind")
    expected = {row["anonymous_id"]: row for row in index["records"]}
    records = payload.get("records", [])
    seen = set()
    for row in records:
        identifier = row.get("anonymous_id")
        if identifier in seen:
            errors.append(f"{identifier}: duplicate record")
            continue
        seen.add(identifier)
        if identifier not in expected:
            errors.append(f"{identifier}: unknown anonymous id")
            continue
        if "goal_language" in row or "outcome" in row or "reward" in row:
            errors.append(f"{identifier}: forbidden information in physical annotation")
        frame_count = int(expected[identifier]["frames"])
        close_frames = [int(value) for value in expected[identifier].get("close_frames", [])]
        annotations = row.get("annotations", [])
        if len(annotations) != len(close_frames):
            errors.append(f"{identifier}: expected {len(close_frames)} close-event annotations, got {len(annotations)}")
        for event in annotations:
            close_frame = event.get("close_frame")
            if close_frame not in close_frames:
                errors.append(f"{identifier}: close_frame {close_frame} not in manifest")
            if event.get("contact_visibility") not in VISIBILITY:
                errors.append(f"{identifier}: invalid contact_visibility")
            if event.get("contact_entity_type") not in ENTITY_TYPES:
                errors.append(f"{identifier}: invalid contact_entity_type")
            if event.get("physical_mechanism") not in MECHANISMS:
                errors.append(f"{identifier}: invalid physical_mechanism")
            if event.get("evidence_quality") not in QUALITY:
                errors.append(f"{identifier}: invalid evidence_quality")
            for key in ("contact_onset_frame", "independent_motion_start", "independent_motion_end",
                        "control_start", "control_end"):
                value = event.get(key)
                if value is not None and (not isinstance(value, int) or not 0 <= value < frame_count):
                    errors.append(f"{identifier}: {key} outside [0, {frame_count})")
            for start, end in (("independent_motion_start", "independent_motion_end"),
                               ("control_start", "control_end")):
                a, b = event.get(start), event.get(end)
                if (a is None) != (b is None):
                    errors.append(f"{identifier}: {start}/{end} must both be null or both present")
                elif a is not None and a > b:
                    errors.append(f"{identifier}: {start} exceeds {end}")
    missing = sorted(set(expected) - seen)
    if missing:
        errors.append(f"missing records: {missing}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    args = parser.parse_args()
    errors = validate(
        json.loads(args.annotations.read_text(encoding="utf-8")),
        json.loads(args.index.read_text(encoding="utf-8")),
    )
    if errors:
        print(json.dumps({"valid": False, "errors": errors}, ensure_ascii=False, indent=2))
        raise SystemExit(1)
    print(json.dumps({"valid": True}, indent=2))


if __name__ == "__main__":
    main()
