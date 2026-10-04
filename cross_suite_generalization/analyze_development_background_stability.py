#!/usr/bin/env python3
"""Audit external-camera background stability on development episodes only."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vla_supervisor.background_motion import estimate_background_motion


def percentile(values: list[float]) -> dict:
    if not values:
        return {}
    return {f"p{q:02d}": float(np.percentile(values, q)) for q in (5, 25, 50, 75, 95)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=5)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    pilot = json.loads(args.pilot.read_text(encoding="utf-8"))
    first_by_goal = {}
    for job in manifest["jobs"]:
        first_by_goal.setdefault(job["goal_language"], job)

    episodes = []
    all_rows = []
    for pilot_episode in pilot["episodes"]:
        goal = pilot_episode["goal_language"]
        job = first_by_goal[goal]
        with np.load(job["sidecar"], mmap_mode="r") as sidecar:
            frames = np.asarray(sidecar["agent_images"], dtype=np.uint8)
        rows = []
        for frame_index in range(0, len(frames) - 1, args.stride):
            estimate = estimate_background_motion(frames[frame_index], frames[frame_index + 1])
            center_displacement = None
            if estimate.homography is not None:
                height, width = frames.shape[1:3]
                center = np.asarray([[[width / 2, height / 2]]], dtype=np.float32)
                projected = cv2.perspectiveTransform(center, estimate.homography)
                center_displacement = float(np.linalg.norm(projected - center))
            row = {
                "frame_index": frame_index,
                "valid": estimate.valid,
                "tracked_points": estimate.tracked_points,
                "inlier_ratio": estimate.inlier_ratio,
                "median_reprojection_error_px": estimate.median_reprojection_error_px,
                "spatial_coverage": estimate.spatial_coverage,
                "confidence": estimate.confidence,
                "predicted_center_background_displacement_px": center_displacement,
            }
            rows.append(row)
            all_rows.append(row)
        episodes.append({
            "goal_language": goal,
            "frames": len(frames),
            "sampled_pairs": len(rows),
            "valid_fraction": sum(row["valid"] for row in rows) / max(len(rows), 1),
        })

    valid = [row for row in all_rows if row["valid"]]
    result = {
        "schema_version": 1,
        "scope": "five development episodes only; frozen holdout not accessed",
        "camera": "agent external view",
        "stride": args.stride,
        "sampled_pairs": len(all_rows),
        "valid_pairs": len(valid),
        "valid_fraction": len(valid) / max(len(all_rows), 1),
        "inlier_ratio": percentile([row["inlier_ratio"] for row in valid]),
        "median_reprojection_error_px": percentile([row["median_reprojection_error_px"] for row in valid]),
        "spatial_coverage": percentile([row["spatial_coverage"] for row in valid]),
        "confidence": percentile([row["confidence"] for row in valid]),
        "predicted_center_background_displacement_px": percentile([
            row["predicted_center_background_displacement_px"] for row in valid
            if row["predicted_center_background_displacement_px"] is not None
        ]),
        "episodes": episodes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "episodes"}, indent=2))


if __name__ == "__main__":
    main()
