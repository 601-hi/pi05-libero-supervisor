#!/usr/bin/env python3
"""Search interpretable post-grasp temporal scores on consumed Wave 1 only."""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np


def displacement(points: np.ndarray, start: int, stop: int, diagonal: float) -> float:
    segment = points[max(0, start) : min(len(points), stop)]
    finite = np.all(np.isfinite(segment), axis=1)
    if not finite.any():
        return 0.0
    valid = segment[finite]
    return float(np.max(np.linalg.norm(valid - valid[0], axis=1)) / diagonal)


def event_values(points: np.ndarray, close: int, diagonal: float) -> dict:
    return {
        "pre10": displacement(points, close - 10, close + 1, diagonal),
        "pre20": displacement(points, close - 20, close + 1, diagonal),
        "post20": displacement(points, close, close + 21, diagonal),
        "post40": displacement(points, close, close + 41, diagonal),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    predictions = json.loads(args.predictions.read_text(encoding="utf-8"))["records"]
    feature_rows = json.loads(args.features.read_text(encoding="utf-8"))["records"]
    annotations = json.loads(args.annotations.read_text(encoding="utf-8"))["records"]
    features = {row["anonymous_id"]: row for row in feature_rows}
    truth = {row["anonymous_id"]: set(row["acceptable_manipulated_candidate_ids"]) for row in annotations}

    coefficients = list(itertools.product((0.0, 0.5, 1.0, 2.0, 4.0), (0.0, 0.25, 0.5, 1.0, 2.0)))
    results = []
    for pre_weight, tail_penalty in coefficients:
        correct = 0
        margins = []
        episode_predictions = []
        for row in predictions:
            episode = features[row["anonymous_id"]]
            diagonal = float(np.hypot(224, 224))
            scores = {}
            for candidate_id, raw_points in row["centroids_xy"].items():
                points = np.asarray([[np.nan, np.nan] if point is None else point for point in raw_points], dtype=float)
                candidates = []
                for close in episode["close_frames"]:
                    if len(points) - close < 12:
                        continue
                    values = event_values(points, close, diagonal)
                    remaining_fraction = (len(points) - close) / len(points)
                    score = values["post40"] - pre_weight * values["pre20"] - tail_penalty * max(0.0, 0.2 - remaining_fraction)
                    candidates.append(score)
                scores[int(candidate_id)] = max(candidates, default=-np.inf)
            ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
            predicted = ranked[0][0] if ranked and np.isfinite(ranked[0][1]) else None
            margin = ranked[0][1] - ranked[1][1] if len(ranked) > 1 else 0.0
            correct += predicted in truth[row["anonymous_id"]]
            margins.append(float(margin))
            episode_predictions.append({"anonymous_id": row["anonymous_id"], "predicted": predicted, "margin": float(margin)})
        results.append({
            "pre_weight": pre_weight,
            "tail_penalty": tail_penalty,
            "correct": correct,
            "median_margin": float(np.median(margins)),
            "predictions": episode_predictions,
        })
    results.sort(key=lambda row: (-row["correct"], -row["median_margin"], row["pre_weight"], row["tail_penalty"]))
    args.output.write_text(json.dumps({"schema_version": 1, "records": results}, indent=2) + "\n", encoding="utf-8")
    for row in results[:10]:
        print({key: row[key] for key in ("pre_weight", "tail_penalty", "correct", "median_margin")})


if __name__ == "__main__":
    main()
