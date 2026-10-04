#!/usr/bin/env python3
"""Fuse fixed motion, wrist candidate quality, and wrist stability without semantics."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixed-motion", type=Path, required=True)
    parser.add_argument("--wrist-scores", type=Path, required=True)
    parser.add_argument("--wrist-stability", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    fixed = {row["anonymous_id"]: row for row in json.loads(args.fixed_motion.read_text(encoding="utf-8"))["records"]}
    scores = {row["anonymous_id"]: row for row in json.loads(args.wrist_scores.read_text(encoding="utf-8"))["records"]}
    stability = {row["anonymous_id"]: row for row in json.loads(args.wrist_stability.read_text(encoding="utf-8"))["records"]}
    rows = []
    for identifier, score in scores.items():
        fixed_groups = fixed[identifier]["events"][0]["co_motion_groups"] if fixed[identifier]["events"] else []
        candidate_high = bool(score.get("passes_frozen_candidate_gate"))
        stable = stability[identifier].get("attachment_established_frame") is not None
        if not fixed_groups:
            state = "no_fixed_motion_evidence"
        elif not candidate_high:
            state = "wrist_candidate_abstained"
        elif not stable:
            state = "wrist_stability_not_established"
        elif len(fixed_groups) > 1:
            state = "possible_control_crossview_ambiguous"
        else:
            state = "possible_control"
        rows.append({
            "anonymous_id": identifier,
            "state": state,
            "fixed_motion_groups": fixed_groups,
            "wrist_candidate_set": score.get("candidate_set", []),
            "controlled_object_candidate_probability": score.get("controlled_object_candidate_probability"),
            "wrist_stability_established": stable,
        })
    result = {
        "schema_version": 1,
        "interpretation": "possible_control is physical support only; not target identity or task success",
        "forbidden_claims": [
            "fixed and wrist candidates have proven shared identity",
            "possible_control means the intended object is controlled",
            "possible_control means task success",
        ],
        "state_counts": dict(Counter(row["state"] for row in rows)),
        "records": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["state_counts"], indent=2))


if __name__ == "__main__":
    main()
