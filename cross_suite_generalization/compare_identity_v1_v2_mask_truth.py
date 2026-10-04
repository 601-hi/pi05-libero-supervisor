#!/usr/bin/env python3
"""Paired v1/v2 comparison against validated initial-mask identity truth."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path


def exact_mcnemar_two_sided(improved: int, regressed: int) -> float:
    n = improved + regressed
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, index) for index in range(min(improved, regressed) + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if total == 0:
        return [0.0, 1.0]
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [center - half, center + half]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v1", type=Path, required=True)
    parser.add_argument("--v2", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    v1 = {r["anonymous_id"]: r for r in json.loads(args.v1.read_text(encoding="utf-8"))["records"]}
    v2 = {r["anonymous_id"]: r for r in json.loads(args.v2.read_text(encoding="utf-8"))["records"]}
    truth = json.loads(args.annotations.read_text(encoding="utf-8"))["records"]
    rows, matrix = [], Counter()
    for label in truth:
        identifier = label["anonymous_id"]
        if not label["target_mask_observable"]:
            rows.append({"anonymous_id": identifier, "v1": "unobservable", "v2": "unobservable"})
            continue
        acceptable = set(label["acceptable_target_mask_ids"])
        v1_choice = v1[identifier]["locked_candidate_id"]
        v1_status = "unknown" if v1_choice is None else ("correct" if v1_choice in acceptable else "wrong")
        v2_status = v2[identifier]["evaluation"]
        v2_choice = None if v2[identifier]["first_lock"] is None else v2[identifier]["first_lock"]["candidate_id"]
        matrix[(v1_status, v2_status)] += 1
        rows.append({"anonymous_id": identifier, "annotation_confidence": label["confidence"],
                     "acceptable_ids": sorted(acceptable), "v1_choice": v1_choice, "v1": v1_status,
                     "v2_choice": v2_choice, "v2": v2_status})
    observable = [row for row in rows if row["v1"] != "unobservable"]
    v1_correct = sum(row["v1"] == "correct" for row in observable)
    v2_correct = sum(row["v2"] == "correct" for row in observable)
    improved = sum(row["v1"] != "correct" and row["v2"] == "correct" for row in observable)
    regressed = sum(row["v1"] == "correct" and row["v2"] != "correct" for row in observable)
    result = {
        "schema_version": 1,
        "observable_episodes": len(observable),
        "unobservable_episodes": len(rows) - len(observable),
        "v1": {"correct": v1_correct, "accuracy": v1_correct / len(observable), "wilson95": wilson(v1_correct, len(observable))},
        "v2": {"correct": v2_correct, "accuracy": v2_correct / len(observable), "wilson95": wilson(v2_correct, len(observable))},
        "paired": {"improved": improved, "regressed": regressed,
                   "exact_two_sided_mcnemar_p": exact_mcnemar_two_sided(improved, regressed),
                   "matrix": {f"{left}->{right}": count for (left, right), count in sorted(matrix.items())}},
        "records": rows,
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("observable_episodes", "unobservable_episodes", "v1", "v2", "paired")}, indent=2))


if __name__ == "__main__":
    main()
