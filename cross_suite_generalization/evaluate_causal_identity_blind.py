#!/usr/bin/env python3
"""Evaluate frozen identity predictions against sealed blind annotations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    predicted = json.loads(args.predictions.read_text(encoding="utf-8"))["records"]
    annotated = json.loads(args.annotations.read_text(encoding="utf-8"))["records"]
    annotations = {row["anonymous_id"]: row for row in annotated}
    if [row["anonymous_id"] for row in predicted] != [row["anonymous_id"] for row in annotated]:
        raise ValueError("Prediction and annotation IDs or ordering differ")

    details = []
    for row in predicted:
        truth = annotations[row["anonymous_id"]]
        locked = row["locked_candidate_id"]
        acceptable = truth["acceptable_manipulated_candidate_ids"]
        status = "unknown" if locked is None else ("correct" if locked in acceptable else "wrong")
        details.append({
            "anonymous_id": row["anonymous_id"],
            "status": status,
            "locked_candidate_id": locked,
            "acceptable_manipulated_candidate_ids": acceptable,
            "lock_frame": row["lock_frame"],
            "num_frames": row["num_frames"],
            "lock_fraction": None if row["lock_frame"] is None else row["lock_frame"] / max(row["num_frames"] - 1, 1),
            "annotation_confidence": truth["confidence"],
        })

    counts = {name: sum(row["status"] == name for row in details) for name in ("correct", "wrong", "unknown")}
    observable = sum(bool(row["identity_observable"]) for row in annotated)
    lock_fractions = [row["lock_fraction"] for row in details if row["lock_fraction"] is not None]
    summary = {
        "episodes": len(details),
        "observable_episodes": observable,
        **counts,
        "selective_accuracy_when_locked": counts["correct"] / max(counts["correct"] + counts["wrong"], 1),
        "coverage": (counts["correct"] + counts["wrong"]) / max(observable, 1),
        "overall_correct_rate": counts["correct"] / max(observable, 1),
        "lock_fraction_median": float(np.median(lock_fractions)) if lock_fractions else None,
        "lock_fraction_p90": float(np.percentile(lock_fractions, 90)) if lock_fractions else None,
    }
    result = {"schema_version": 1, "summary": summary, "records": details}
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    wrong = [row for row in details if row["status"] != "correct"]
    print("NON_CORRECT", json.dumps(wrong, indent=2))


if __name__ == "__main__":
    main()
