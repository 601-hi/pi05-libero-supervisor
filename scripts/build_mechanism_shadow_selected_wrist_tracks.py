#!/usr/bin/env python3
"""Build short post-close tracks only for frozen high-quality unique candidates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bidirectional", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    tracks = {
        row["anonymous_id"]: row
        for row in json.loads(args.bidirectional.read_text(encoding="utf-8"))["records"]
    }
    scores = json.loads(args.scores.read_text(encoding="utf-8"))["records"]
    records = []
    for score in scores:
        identifier = score["anonymous_id"]
        if not score.get("passes_frozen_candidate_gate"):
            records.append({
                "anonymous_id": identifier,
                "status": "candidate_gate_abstained",
                "selection_state": score["state"],
            })
            continue
        candidate_set = score["candidate_set"]
        if len(candidate_set) != 1:
            raise ValueError(f"{identifier}: passed gate without a unique candidate")
        candidate_id = int(candidate_set[0])
        source = tracks[identifier]
        key = str(candidate_id)
        centroids = source["post_centroids_xy"][key]
        areas = source["post_area_fraction"][key]
        if len(centroids) != len(areas):
            raise ValueError(f"{identifier}: centroid/area length mismatch")
        records.append({
            "anonymous_id": identifier,
            "status": "tracked",
            "close_frame": source["close_frame"],
            "tracked_frames": len(centroids),
            "source_wrist_candidate_id": candidate_id,
            "candidate_quality_probability": score["controlled_object_candidate_probability"],
            "centroids_xy": centroids,
            "area_fraction": areas,
        })
    result = {
        "schema_version": 1,
        "role": "short causal-window wrist stability input",
        "outcome_labels_used": False,
        "selection_rule": "unique candidate above frozen controlled-object candidate gate",
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "episodes": len(records),
        "tracked": sum(row["status"] == "tracked" for row in records),
        "abstained": sum(row["status"] != "tracked" for row in records),
    }, indent=2))


if __name__ == "__main__":
    main()
