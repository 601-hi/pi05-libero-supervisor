#!/usr/bin/env python3
"""Intersect fixed-to-wrist appearance and wrist physical candidate sets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def consensus(appearance: dict | None, physical_candidates: list[dict], appearance_margin: float, physical_margin: float) -> dict:
    if appearance is None or not physical_candidates:
        return {"state": "unavailable", "candidate_ids": []}
    appearance_rows = appearance["candidate_scores"]
    best_appearance = max(row["appearance"] for row in appearance_rows)
    appearance_ids = {
        int(row["candidate_id"]) for row in appearance_rows
        if row["appearance"] >= best_appearance - appearance_margin
    }
    best_physical = max(row["physical_fusion_score"] for row in physical_candidates)
    physical_ids = {
        int(row["candidate_id"]) for row in physical_candidates
        if row["physical_fusion_score"] >= best_physical - physical_margin
    }
    intersection = sorted(appearance_ids & physical_ids)
    state = "unique" if len(intersection) == 1 else "ambiguous" if intersection else "conflict"
    return {
        "state": state,
        "candidate_ids": intersection,
        "appearance_candidate_ids": sorted(appearance_ids),
        "physical_candidate_ids": sorted(physical_ids),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bridge", type=Path, required=True)
    parser.add_argument("--physical-fusion", type=Path, required=True)
    parser.add_argument("--wrist-annotations", type=Path, required=True)
    parser.add_argument("--fixed-annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--bridge-field", choices=["set_pipeline_bridge", "physical_ranked_pipeline_bridge"],
        default="set_pipeline_bridge",
    )
    parser.add_argument("--appearance-margin", type=float, default=.03)
    parser.add_argument("--physical-margin", type=float, default=.10)
    args = parser.parse_args()
    bridge = json.loads(args.bridge.read_text(encoding="utf-8"))["records"]
    physical = {row["anonymous_id"]: row for row in json.loads(args.physical_fusion.read_text(encoding="utf-8"))["episodes"]}
    wrist_truth = {
        key: set(value["acceptable_candidate_ids"])
        for key, value in json.loads(args.wrist_annotations.read_text(encoding="utf-8"))["annotations"].items()
    }
    fixed_truth = {
        row["anonymous_id"]: set(row["acceptable_target_mask_ids"])
        for row in json.loads(args.fixed_annotations.read_text(encoding="utf-8"))["records"]
    }
    rows = []
    for record in bridge:
        identifier = record["anonymous_id"]
        appearance = record.get(args.bridge_field)
        result = consensus(appearance, physical[identifier]["candidates"], args.appearance_margin, args.physical_margin)
        chosen = result["candidate_ids"][0] if result["state"] == "unique" else None
        matched_fixed = None
        if chosen is not None:
            score_row = next(row for row in appearance["candidate_scores"] if int(row["candidate_id"]) == chosen)
            matched_fixed = int(score_row["matched_fixed_id"])
        rows.append({
            "anonymous_id": identifier,
            **result,
            "chosen_wrist_correct": None if chosen is None else chosen in wrist_truth[identifier],
            "chosen_pair_correct": None if chosen is None else (
                chosen in wrist_truth[identifier] and matched_fixed in fixed_truth[identifier]
            ),
        })
    unique = [row for row in rows if row["state"] == "unique"]
    output = {
        "schema_version": 1,
        "evaluation_role": "development/retrospective is determined by supplied artifacts",
        "appearance_margin": args.appearance_margin,
        "physical_margin": args.physical_margin,
        "bridge_field": args.bridge_field,
        "summary": {
            "episodes": len(rows),
            "unique": len(unique),
            "ambiguous": sum(row["state"] == "ambiguous" for row in rows),
            "conflict": sum(row["state"] == "conflict" for row in rows),
            "unavailable": sum(row["state"] == "unavailable" for row in rows),
            "unique_wrist_correct": sum(bool(row["chosen_wrist_correct"]) for row in unique),
            "unique_pair_correct": sum(bool(row["chosen_pair_correct"]) for row in unique),
        },
        "records": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output["summary"], indent=2))


if __name__ == "__main__":
    main()
