"""Evaluate a small set-valued fixed-view motion proposal rule."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def proposal_ids(replay: dict, feature: dict) -> set[int]:
    """Union v2 first lock with the candidate having largest net displacement."""
    candidates = feature["candidates"]
    if not candidates:
        return set()
    displacement_id = max(
        candidates,
        key=lambda key: float(candidates[key].get("maximum_net_displacement") or float("-inf")),
    )
    result = {int(displacement_id)}
    first_lock = replay.get("first_lock")
    if first_lock is not None:
        result.add(int(first_lock["candidate_id"]))
    return result


def evaluate(replay_records: list[dict], feature_records: list[dict], annotations: list[dict]) -> dict:
    replay = {row["anonymous_id"]: row for row in replay_records}
    features = {row["anonymous_id"]: row for row in feature_records}
    rows = []
    for annotation in annotations:
        identifier = annotation["anonymous_id"]
        observable = bool(annotation.get("target_mask_observable", True))
        truth = set(map(int, annotation.get("acceptable_target_mask_ids", [])))
        proposal = proposal_ids(replay[identifier], features[identifier])
        rows.append(
            {
                "anonymous_id": identifier,
                "observable": observable,
                "proposal_ids": sorted(proposal),
                "truth_ids": sorted(truth),
                "hit": bool(observable and proposal.intersection(truth)),
                "proposal_size": len(proposal),
            }
        )
    observable_rows = [row for row in rows if row["observable"]]
    hits = sum(row["hit"] for row in observable_rows)
    sizes = [row["proposal_size"] for row in observable_rows]
    return {
        "summary": {
            "episodes": len(rows),
            "observable_episodes": len(observable_rows),
            "hits": hits,
            "recall": hits / len(observable_rows) if observable_rows else None,
            "average_proposal_size": sum(sizes) / len(sizes) if sizes else None,
            "maximum_proposal_size": max(sizes) if sizes else None,
            "size_two_episodes": sum(size == 2 for size in sizes),
        },
        "records": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    replay = json.loads(args.replay.read_text(encoding="utf-8"))["records"]
    features = json.loads(args.features.read_text(encoding="utf-8"))["records"]
    annotations = json.loads(args.annotations.read_text(encoding="utf-8"))["records"]
    result = {
        "schema_version": 1,
        "rule": "union of v2 first lock and maximum-net-displacement candidate",
        "interpretation_limit": "Set recall of the annotated complete object; not proposal precision or background-homography accuracy.",
        **evaluate(replay, features, annotations),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
