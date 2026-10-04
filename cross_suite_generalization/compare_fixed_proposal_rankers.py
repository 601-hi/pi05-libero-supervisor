"""Compare label-free physical rankers inside a small fixed-view proposal set.

The script is an audit tool, not a learned classifier.  Candidate proposals are
fixed upstream.  Wave labels are used only to count whether each explicit
physical ranking rule retains the complete manipulated object.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def event_max(candidate: dict, key: str, default: float = float("-inf")) -> float:
    values = [event.get(key) for event in candidate.get("events", [])]
    values = [float(value) for value in values if value is not None]
    return max(values) if values else default


def event_min(candidate: dict, key: str, default: float = float("inf")) -> float:
    values = [event.get(key) for event in candidate.get("events", [])]
    values = [float(value) for value in values if value is not None]
    return min(values) if values else default


def score(candidate: dict, rule: str) -> float:
    if rule == "maximum_net_displacement":
        return float(candidate.get("maximum_net_displacement") or 0.0)
    if rule == "post_close_max_displacement":
        return event_max(candidate, "post40_max_displacement", 0.0)
    if rule == "post_close_residual_sum":
        return event_max(candidate, "post40_residual_sum", 0.0)
    if rule == "post_over_pre_residual":
        post = event_max(candidate, "post40_background_residual_median", 0.0)
        pre = event_max(candidate, "pre20_background_residual_median", 0.0)
        return post / max(pre, 1e-9)
    if rule == "earliest_post_close_onset":
        onset = event_min(candidate, "post40_displacement_onset_001")
        return -onset if onset != float("inf") else float("-inf")
    raise ValueError(rule)


RULES = (
    "maximum_net_displacement",
    "post_close_max_displacement",
    "post_close_residual_sum",
    "post_over_pre_residual",
    "earliest_post_close_onset",
)


def evaluate(features: list[dict], proposals: list[dict], annotations: list[dict]) -> dict:
    feature_by_id = {row["anonymous_id"]: row for row in features}
    proposal_by_id = {row["anonymous_id"]: row for row in proposals}
    rows = []
    for annotation in annotations:
        identifier = annotation["anonymous_id"]
        truth = set(map(int, annotation.get("acceptable_target_mask_ids", [])))
        observable = bool(annotation.get("target_mask_observable", True)) and bool(truth)
        proposal = set(map(int, proposal_by_id[identifier]["proposal_ids"]))
        candidates = feature_by_id[identifier]["candidates"]
        selected = {}
        for rule in RULES:
            selected[rule] = max(
                proposal,
                key=lambda candidate_id: (score(candidates[str(candidate_id)], rule), -candidate_id),
            ) if proposal else None
        rows.append({
            "anonymous_id": identifier,
            "observable": observable,
            "truth_ids": sorted(truth),
            "proposal_ids": sorted(proposal),
            "proposal_contains_truth": bool(observable and proposal.intersection(truth)),
            "selected_ids": selected,
            "correct": {
                rule: bool(observable and candidate_id in truth)
                for rule, candidate_id in selected.items()
            },
        })
    eligible = [row for row in rows if row["proposal_contains_truth"]]
    return {
        "schema_version": 1,
        "interpretation": (
            "Conditional physical ranking accuracy when the frozen proposal set contains the "
            "annotated complete object; labels never enter any score."
        ),
        "summary": {
            "episodes": len(rows),
            "proposal_contains_truth": len(eligible),
            "conditional_accuracy": {
                rule: sum(row["correct"][rule] for row in eligible) / len(eligible) if eligible else None
                for rule in RULES
            },
            "correct_counts": {
                rule: sum(row["correct"][rule] for row in eligible) for rule in RULES
            },
        },
        "records": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--proposals", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(
        json.loads(args.features.read_text(encoding="utf-8"))["records"],
        json.loads(args.proposals.read_text(encoding="utf-8"))["records"],
        json.loads(args.annotations.read_text(encoding="utf-8"))["records"],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
