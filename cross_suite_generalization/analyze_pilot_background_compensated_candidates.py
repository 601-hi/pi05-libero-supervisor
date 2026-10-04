#!/usr/bin/env python3
"""Rank development candidates by centroid motion after robust background compensation."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vla_supervisor.background_motion import estimate_background_motion


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--review-grid", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    pilot = json.loads(args.pilot.read_text(encoding="utf-8"))
    review = json.loads(args.review_grid.read_text(encoding="utf-8"))
    expected = {
        row["goal_index"]: row["expected_manipulated_id_from_blind_review"]
        for row in review["configurations"][0]["episodes"]
    }
    first_by_goal = {}
    for job in manifest["jobs"]:
        first_by_goal.setdefault(job["goal_language"], job)

    results = []
    for episode in pilot["episodes"]:
        with np.load(first_by_goal[episode["goal_language"]]["sidecar"], mmap_mode="r") as sidecar:
            frames = np.asarray(sidecar["agent_images"], dtype=np.uint8)
        estimates = [estimate_background_motion(frames[t], frames[t + 1]) for t in range(len(frames) - 1)]
        candidate_results = []
        diagonal = float(np.hypot(*frames.shape[1:3]))
        for candidate in episode["motion_ranked_candidates"]:
            points = candidate["centroid_xy"]
            residuals = []
            for t, estimate in enumerate(estimates):
                if not estimate.valid or estimate.homography is None or points[t] is None or points[t + 1] is None:
                    continue
                source = np.asarray([[points[t]]], dtype=np.float32)
                predicted = cv2.perspectiveTransform(source, estimate.homography)[0, 0]
                residuals.append(float(np.linalg.norm(np.asarray(points[t + 1]) - predicted)))
            values = np.asarray(residuals, dtype=float)
            candidate_results.append({
                "object_id": candidate["object_id"],
                "valid_transitions": len(values),
                "background_compensated_path_normalized": float(values.sum() / diagonal),
                "median_residual_px": float(np.median(values)) if len(values) else None,
                "p90_residual_px": float(np.percentile(values, 90)) if len(values) else None,
                "fraction_residual_gt_1p5px": float(np.mean(values > 1.5)) if len(values) else None,
                "original_centroid_path_normalized": candidate["centroid_path_normalized"],
            })
        candidate_results.sort(key=lambda row: row["background_compensated_path_normalized"], reverse=True)
        expected_id = expected[episode["goal_index"]]
        rank = next((index + 1 for index, row in enumerate(candidate_results) if row["object_id"] == expected_id), None)
        results.append({
            "goal_index": episode["goal_index"],
            "goal_language": episode["goal_language"],
            "expected_id_from_identity_review": expected_id,
            "expected_background_compensated_rank": rank,
            "valid_background_fraction": sum(item.valid for item in estimates) / max(len(estimates), 1),
            "candidates": candidate_results,
        })
    output = {
        "schema_version": 1,
        "scope": "five development episodes only; no frozen holdout access",
        "metric": "sum of per-frame centroid residual after majority-background homography, normalized by image diagonal",
        "episodes": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "episodes": len(results),
        "expected_ranks": {str(row["goal_index"]): row["expected_background_compensated_rank"] for row in results},
        "rank1": sum(row["expected_background_compensated_rank"] == 1 for row in results),
    }, indent=2))


if __name__ == "__main__":
    main()
