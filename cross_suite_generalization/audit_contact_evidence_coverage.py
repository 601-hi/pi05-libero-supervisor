"""Audit whether existing artifacts can drive contact/grasp diagnosis honestly.

The audit never converts uncalibrated geometric residuals into probabilities
and never reads episode outcomes.  It reports availability, temporal coverage,
and whether every field is ready for a frozen prospective replay.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


FIELDS = {
    "execution_four_state": "calibrated",
    "background_confidence": "calibrated",
    "gripper_candidate_proximity": "missing_calibrated_probability",
    "independent_motion_probability": "raw_geometry_only",
    "attachment_probability": "thresholded_events_only",
    "blocked_motion_probability": "missing_calibrated_probability",
    "semantic_target_probability": "missing_calibrated_probability",
}


def load_records(path: Path) -> dict[str, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {row["anonymous_id"]: row for row in payload["records"]}


def audit(background: dict[str, dict], wrist: dict[str, dict], alignment: dict[str, dict]) -> dict:
    identifiers = sorted(set(background) | set(wrist) | set(alignment))
    rows = []
    for identifier in identifiers:
        bg = background.get(identifier)
        wr = wrist.get(identifier)
        action = alignment.get(identifier)
        bg_sequence = [] if bg is None else bg.get("background_confidence_sequence", [])
        wrist_events = [] if wr is None else wr.get("events", [])
        fields = {
            "execution_four_state": bool(action and action.get("scored_steps", 0) > 0),
            "background_confidence": bool(bg_sequence),
            # Existing artifacts do not contain these as calibrated per-step probabilities.
            "gripper_candidate_proximity": False,
            "independent_motion_probability": False,
            "attachment_probability": False,
            "blocked_motion_probability": False,
            "semantic_target_probability": False,
        }
        rows.append({
            "anonymous_id": identifier,
            "available_fields": fields,
            "background_frames": len(bg_sequence),
            "execution_scored_steps": 0 if action is None else int(action.get("scored_steps", 0)),
            "wrist_thresholded_events": len(wrist_events),
            "attachment_established_event_available": any(
                event.get("type") == "attachment_established" for event in wrist_events
            ),
            "prospective_contact_replay_ready": all(fields.values()),
        })
    count = len(rows)
    coverage = {
        field: sum(row["available_fields"][field] for row in rows) / count if count else 0.0
        for field in FIELDS
    }
    return {
        "schema_version": 1,
        "audit_role": "availability only; no outcomes or correctness labels used",
        "probability_policy": (
            "Raw background residuals and thresholded wrist events are not silently converted "
            "into calibrated per-step probabilities. Missing fields remain missing."
        ),
        "field_status": FIELDS,
        "summary": {
            "episodes": count,
            "field_episode_coverage": coverage,
            "episodes_ready_for_full_contact_replay": sum(
                row["prospective_contact_replay_ready"] for row in rows
            ),
            "episodes_with_attachment_established_event": sum(
                row["attachment_established_event_available"] for row in rows
            ),
        },
        "records": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--background", type=Path, required=True)
    parser.add_argument("--wrist", type=Path, required=True)
    parser.add_argument("--alignment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(
        load_records(args.background),
        load_records(args.wrist),
        load_records(args.alignment),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
