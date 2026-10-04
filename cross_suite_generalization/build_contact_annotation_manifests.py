"""Build leakage-resistant physical and semantic annotation manifests."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path, wave: str) -> list[dict]:
    records = json.loads(path.read_text(encoding="utf-8"))["records"]
    result = []
    for row in records:
        frame_count = len(row.get("background_confidence_sequence", []))
        result.append({
            "anonymous_id": row["anonymous_id"],
            "development_wave": wave,
            "frame_count": frame_count,
            "close_frames": [int(value) for value in row.get("close_frames", [])],
            "goal_language": row.get("goal_language"),
        })
    return result


def build(records: list[dict]) -> tuple[dict, dict]:
    identifiers = [row["anonymous_id"] for row in records]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("anonymous ids overlap across development waves")
    physical = {
        "schema_version": 1,
        "annotation_pass": "A_physical_blind",
        "forbidden_information": [
            "goal language", "episode outcome", "reward", "execution expert score",
            "model-selected candidate identity", "semantic-model output",
        ],
        "records": [{
            "anonymous_id": row["anonymous_id"],
            "development_wave": row["development_wave"],
            "frame_count": row["frame_count"],
            "close_frames": row["close_frames"],
            "annotations": [{
                "close_frame": close_frame,
                "contact_onset_frame": None,
                "contact_visibility": None,
                "contact_entity_type": None,
                "independent_motion_start": None,
                "independent_motion_end": None,
                "control_start": None,
                "control_end": None,
                "physical_mechanism": None,
                "evidence_quality": None,
                "notes": "",
            } for close_frame in row["close_frames"]],
        } for row in records],
    }
    semantic = {
        "schema_version": 1,
        "annotation_pass": "B_semantic_after_physical_freeze",
        "prerequisite": "Pass A controlled-object identity and event intervals are frozen",
        "forbidden_information": [
            "episode outcome", "reward", "execution expert score", "future task success",
        ],
        "records": [{
            "anonymous_id": row["anonymous_id"],
            "development_wave": row["development_wave"],
            "goal_language": row["goal_language"],
            "physical_annotation_reference": None,
            "controlled_object_matches_goal": None,
            "semantic_confidence": None,
            "notes": "",
        } for row in records],
    }
    return physical, semantic


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wave1", type=Path, required=True)
    parser.add_argument("--wave2", type=Path, required=True)
    parser.add_argument("--physical-output", type=Path, required=True)
    parser.add_argument("--semantic-output", type=Path, required=True)
    args = parser.parse_args()
    physical, semantic = build(load(args.wave1, "Wave1") + load(args.wave2, "Wave2"))
    for path, payload in ((args.physical_output, physical), (args.semantic_output, semantic)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"physical_records": len(physical["records"]),
                      "semantic_records": len(semantic["records"])}, indent=2))


if __name__ == "__main__":
    main()
