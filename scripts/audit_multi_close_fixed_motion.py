"""Outcome-blind fixed-view motion audit for independently tracked close events."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.audit_mechanism_shadow_fixed_motion import groups, summarize


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    document = json.loads(args.predictions.read_text(encoding="utf-8"))
    if document.get("protocol", {}).get("camera_key") != "agent_images":
        raise ValueError("fixed-view audit requires camera_key=agent_images")
    outputs = []
    for record in document["records"]:
        candidates = []
        for candidate_id in record["post_centroids_xy"]:
            candidates.append({
                "candidate_id": candidate_id,
                **summarize(record["pre_centroids_xy"][candidate_id], record["post_centroids_xy"][candidate_id]),
            })
        candidates.sort(key=lambda row: (row["sustained_motion"], row["postclose_max_displacement_px"] or -1), reverse=True)
        sustained = {row["candidate_id"] for row in candidates if row["sustained_motion"]}
        outputs.append({
            "event_id": record["event_id"], "anonymous_id": record["anonymous_id"],
            "close_ordinal": record["close_ordinal"], "close_frame": record["close_frame"],
            "available_post_frames": max((len(v) for v in record["post_centroids_xy"].values()), default=0),
            "co_motion_groups": groups(record["post_centroids_xy"], sustained),
            "sustained_candidate_ids": sorted(sustained), "candidates": candidates,
        })
    result = {
        "schema_version": 1,
        "scope": "outcome-blind fixed-view motion evidence; not object identity",
        "forbidden_inputs": ["task language", "episode outcome", "reward"],
        "summary": {"events": len(outputs),
                    "with_sustained_motion": sum(bool(r["sustained_candidate_ids"]) for r in outputs)},
        "records": outputs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
