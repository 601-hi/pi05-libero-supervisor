"""Audit whether a manipulated object stays stable in a wrist-camera frame.

This script deliberately consumes frozen close-frame annotations.  It must not
change those annotations from the later motion tracks that it evaluates.
Wave2 is a development set, so the output is descriptive evidence rather than
a frozen test-set accuracy claim.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def _finite_xy(values: list[Any], start: int, stop: int) -> tuple[np.ndarray, np.ndarray]:
    sliced = values[start:stop]
    valid = np.asarray([v is not None and len(v) == 2 for v in sliced], dtype=bool)
    xy = np.full((len(sliced), 2), np.nan, dtype=float)
    if valid.any():
        xy[valid] = np.asarray([v for v in sliced if v is not None and len(v) == 2], dtype=float)
    return xy, valid


def candidate_metrics(
    centroids: list[Any],
    areas: list[Any],
    *,
    settle_frames: int,
    evaluation_frames: int,
    image_size: int,
) -> dict[str, float]:
    stop = min(len(centroids), len(areas), settle_frames + evaluation_frames)
    xy, valid = _finite_xy(centroids, settle_frames, stop)
    area = np.asarray(areas[settle_frames:stop], dtype=float)
    area_valid = np.isfinite(area) & (area > 0)
    diag = math.sqrt(2.0) * image_size

    if valid.sum() < 2:
        return {
            "visibility_fraction": float(valid.mean()) if len(valid) else 0.0,
            "centroid_dispersion_p90": float("nan"),
            "step_motion_median": float("nan"),
            "step_motion_p90": float("nan"),
            "net_centroid_drift": float("nan"),
            "log_area_mad": float("nan"),
            "attachment_instability": float("inf"),
        }

    valid_xy = xy[valid]
    centre = np.median(valid_xy, axis=0)
    dispersion = np.linalg.norm(valid_xy - centre, axis=1) / diag

    adjacent = valid[:-1] & valid[1:]
    motion = np.linalg.norm(xy[1:] - xy[:-1], axis=1)[adjacent] / diag
    net_drift = np.linalg.norm(valid_xy[-1] - valid_xy[0]) / diag

    if area_valid.any():
        log_area = np.log(area[area_valid])
        log_area_mad = float(np.median(np.abs(log_area - np.median(log_area))))
    else:
        log_area_mad = float("nan")

    dispersion_p90 = float(np.percentile(dispersion, 90))
    motion_median = float(np.median(motion)) if len(motion) else float("nan")
    motion_p90 = float(np.percentile(motion, 90)) if len(motion) else float("nan")
    visibility = float(valid.mean())
    # A transparent development score: lower means more wrist-relative stability.
    # Visibility is a gate/penalty; it is not allowed to turn a missing track into
    # apparently perfect stability.
    score = (
        dispersion_p90
        + motion_p90
        + 0.25 * log_area_mad
        + 0.25 * (1.0 - visibility)
    )
    return {
        "visibility_fraction": visibility,
        "centroid_dispersion_p90": dispersion_p90,
        "step_motion_median": motion_median,
        "step_motion_p90": motion_p90,
        "net_centroid_drift": float(net_drift),
        "log_area_mad": log_area_mad,
        "attachment_instability": float(score),
    }


def percentile_summary(values: list[float]) -> dict[str, float]:
    finite = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if not len(finite):
        return {"count": 0}
    return {
        "count": int(len(finite)),
        "p10": float(np.percentile(finite, 10)),
        "p25": float(np.percentile(finite, 25)),
        "median": float(np.median(finite)),
        "p75": float(np.percentile(finite, 75)),
        "p90": float(np.percentile(finite, 90)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--settle-frames", type=int, default=5)
    parser.add_argument("--evaluation-frames", type=int, default=25)
    parser.add_argument("--image-size", type=int, default=224)
    args = parser.parse_args()

    predictions = json.loads(args.predictions.read_text(encoding="utf-8"))
    annotation_doc = json.loads(args.annotations.read_text(encoding="utf-8"))
    annotations = annotation_doc["annotations"]

    candidate_rows: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    for record in predictions["records"]:
        anonymous_id = record["anonymous_id"]
        acceptable = set(annotations[anonymous_id]["acceptable_candidate_ids"])
        per_episode: list[dict[str, Any]] = []
        for candidate_key, centroids in record["centroids_xy"].items():
            candidate_id = int(candidate_key)
            metrics = candidate_metrics(
                centroids,
                record["area_fraction"][candidate_key],
                settle_frames=args.settle_frames,
                evaluation_frames=args.evaluation_frames,
                image_size=args.image_size,
            )
            row = {
                "anonymous_id": anonymous_id,
                "candidate_id": candidate_id,
                "is_acceptable_target": candidate_id in acceptable,
                **metrics,
            }
            candidate_rows.append(row)
            per_episode.append(row)

        target = [r for r in per_episode if r["is_acceptable_target"]]
        non_target = [r for r in per_episode if not r["is_acceptable_target"]]
        target_scores = [r["attachment_instability"] for r in target]
        non_target_scores = [r["attachment_instability"] for r in non_target]
        target_score = float(np.median(target_scores))
        best_non_target = float(min(non_target_scores)) if non_target_scores else float("nan")
        sorted_scores = sorted((r["attachment_instability"], r["candidate_id"]) for r in per_episode)
        acceptable_ranks = [i + 1 for i, (_, cid) in enumerate(sorted_scores) if cid in acceptable]
        episode_rows.append(
            {
                "anonymous_id": anonymous_id,
                "goal_language": record["goal_language"],
                "acceptable_candidate_ids": sorted(acceptable),
                "target_instability_median": target_score,
                "best_non_target_instability": best_non_target,
                "target_better_than_best_non_target": target_score < best_non_target,
                "best_acceptable_target_rank": min(acceptable_ranks),
            }
        )

    metric_names = [
        "visibility_fraction",
        "centroid_dispersion_p90",
        "step_motion_median",
        "step_motion_p90",
        "net_centroid_drift",
        "log_area_mad",
        "attachment_instability",
    ]
    target_rows = [r for r in candidate_rows if r["is_acceptable_target"]]
    non_target_rows = [r for r in candidate_rows if not r["is_acceptable_target"]]
    report = {
        "schema_version": 1,
        "interpretation": "Wave2 development-only wrist-relative attachment audit; no threshold is frozen here.",
        "parameters": {
            "settle_frames": args.settle_frames,
            "evaluation_frames": args.evaluation_frames,
            "image_size": args.image_size,
            "lower_attachment_instability_is_more_stable": True,
        },
        "counts": {
            "episodes": len(episode_rows),
            "target_candidates": len(target_rows),
            "non_target_candidates": len(non_target_rows),
        },
        "target_summary": {m: percentile_summary([r[m] for r in target_rows]) for m in metric_names},
        "non_target_summary": {m: percentile_summary([r[m] for r in non_target_rows]) for m in metric_names},
        "episode_ranking": {
            "target_better_than_best_non_target_count": sum(r["target_better_than_best_non_target"] for r in episode_rows),
            "target_rank_1_count": sum(r["best_acceptable_target_rank"] == 1 for r in episode_rows),
            "target_rank_at_most_2_count": sum(r["best_acceptable_target_rank"] <= 2 for r in episode_rows),
        },
        "episodes": episode_rows,
        "candidates": candidate_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({**report["counts"], **report["episode_ranking"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
