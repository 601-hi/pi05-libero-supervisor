#!/usr/bin/env python3
"""Open outcomes only after frozen wrist candidate scores have been produced."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from statistics import median


def summarize(rows: list[dict]) -> dict:
    probabilities = [
        row["controlled_object_candidate_probability"] for row in rows
        if row.get("controlled_object_candidate_probability") is not None
    ]
    return {
        "episodes": len(rows),
        "with_close": sum(row.get("close_frame") is not None for row in rows),
        "state_counts": dict(Counter(row["state"] for row in rows)),
        "unique_above_frozen_candidate_gate": sum(
            bool(row.get("passes_frozen_candidate_gate")) for row in rows
        ),
        "median_unique_candidate_probability": median(probabilities) if probabilities else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--private-map", type=Path, required=True)
    parser.add_argument("--collection-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    score_doc = json.loads(args.scores.read_text(encoding="utf-8"))
    private = json.loads(args.private_map.read_text(encoding="utf-8"))["records"]
    audit = json.loads(args.collection_audit.read_text(encoding="utf-8"))["records"]
    identity_by_id = {row["anonymous_id"]: row["source_identity"] for row in private}
    outcome_by_identity = {row["identity"]: row for row in audit}
    rows = []
    for score in score_doc["records"]:
        identity = identity_by_id[score["anonymous_id"]]
        outcome = outcome_by_identity[identity]
        rows.append({
            **score,
            "source_identity": identity,
            "suite": outcome["suite"],
            "task_id": outcome["task_id"],
            "success": outcome["success"],
        })

    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups["success" if row["success"] else "failure"].append(row)
        groups[f"suite:{row['suite']}"] .append(row)
    result = {
        "schema_version": 1,
        "evaluation_role": "post-hoc descriptive audit; outcomes were withheld from candidate generation and scoring",
        "forbidden_claims": [
            "candidate gate is a failure detector",
            "candidate probability is grasp success probability",
            "candidate probability establishes physical attachment or target identity",
        ],
        "overall": summarize(rows),
        "by_group": {key: summarize(value) for key, value in sorted(groups.items())},
        "records": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"overall": result["overall"], "by_group": result["by_group"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
