"""Evaluate pre/post-close wrist-relative motion change on frozen masks."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


EPS = 1e-5


def _xy(values: list[Any], image_size: int) -> np.ndarray:
    result = np.full((len(values), 2), np.nan, dtype=float)
    for index, value in enumerate(values):
        if value is not None and len(value) == 2:
            result[index] = value
    return result / (math.sqrt(2.0) * image_size)


def motion_features(centroids: list[Any], areas: list[Any], image_size: int) -> dict[str, float]:
    xy = _xy(centroids, image_size)
    valid = np.all(np.isfinite(xy), axis=1)
    adjacent = valid[:-1] & valid[1:]
    speeds = np.linalg.norm(xy[1:] - xy[:-1], axis=1)[adjacent]
    valid_xy = xy[valid]
    if len(valid_xy):
        centre = np.median(valid_xy, axis=0)
        radius = np.linalg.norm(valid_xy - centre, axis=1)
    else:
        radius = np.asarray([], dtype=float)
    area = np.asarray(areas, dtype=float)
    area = area[np.isfinite(area) & (area > 0)]
    area_delta = np.abs(np.diff(np.log(area))) if len(area) > 1 else np.asarray([], dtype=float)
    return {
        "visibility": float(valid.mean()) if len(valid) else 0.0,
        "speed_median": float(np.median(speeds)) if len(speeds) else float("nan"),
        "speed_p90": float(np.percentile(speeds, 90)) if len(speeds) else float("nan"),
        "dispersion_p90": float(np.percentile(radius, 90)) if len(radius) else float("nan"),
        "area_delta_median": float(np.median(area_delta)) if len(area_delta) else float("nan"),
    }


def log_drop(before: float, after: float) -> float:
    return float(np.log((before + EPS) / (after + EPS)))


def auc_higher_is_target(target: list[float], other: list[float]) -> float:
    wins = 0.0
    total = 0
    for left in target:
        for right in other:
            if np.isfinite(left) and np.isfinite(right):
                wins += float(left > right) + 0.5 * float(left == right)
                total += 1
    return wins / total if total else float("nan")


def summary(values: list[float]) -> dict[str, float]:
    values_array = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    return {
        "count": int(len(values_array)),
        "p10": float(np.percentile(values_array, 10)),
        "median": float(np.median(values_array)),
        "p90": float(np.percentile(values_array, 90)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--post-settle", type=int, default=5)
    parser.add_argument("--post-window", type=int, default=15)
    args = parser.parse_args()

    records = json.loads(args.predictions.read_text(encoding="utf-8"))["records"]
    annotations = json.loads(args.annotations.read_text(encoding="utf-8"))["annotations"]
    rows = []
    for record in records:
        identifier = record["anonymous_id"]
        target_ids = set(annotations[identifier]["acceptable_candidate_ids"])
        observable = bool(annotations[identifier].get("observable", bool(target_ids)))
        for candidate_key, pre_centroids in record["pre_centroids_xy"].items():
            candidate_id = int(candidate_key)
            # Exclude the shared close frame from pre evidence.  Post evidence
            # starts after a short mechanical/segmentation settling interval.
            pre = motion_features(
                pre_centroids[:-1], record["pre_area_fraction"][candidate_key][:-1], args.image_size
            )
            start = args.post_settle
            stop = start + args.post_window
            post = motion_features(
                record["post_centroids_xy"][candidate_key][start:stop],
                record["post_area_fraction"][candidate_key][start:stop],
                args.image_size,
            )
            speed_drop = log_drop(pre["speed_median"], post["speed_median"])
            dispersion_drop = log_drop(pre["dispersion_p90"], post["dispersion_p90"])
            area_drop = log_drop(pre["area_delta_median"], post["area_delta_median"])
            rows.append(
                {
                    "anonymous_id": identifier,
                    "candidate_id": candidate_id,
                    "is_acceptable_target": candidate_id in target_ids,
                    "target_observable": observable,
                    "pre": pre,
                    "post": post,
                    "speed_log_drop": speed_drop,
                    "dispersion_log_drop": dispersion_drop,
                    "area_change_log_drop": area_drop,
                    "combined_change_score": speed_drop + 0.5 * dispersion_drop + 0.25 * area_drop,
                }
            )

    score_names = ["speed_log_drop", "dispersion_log_drop", "area_change_log_drop", "combined_change_score"]
    target = [r for r in rows if r["is_acceptable_target"]]
    other = [r for r in rows if not r["is_acceptable_target"]]
    metrics = {}
    episode_rankings = {}
    for name in score_names:
        target_values = [r[name] for r in target]
        other_values = [r[name] for r in other]
        rank_rows = []
        for record in records:
            identifier = record["anonymous_id"]
            candidates = [r for r in rows if r["anonymous_id"] == identifier]
            ordered = sorted(candidates, key=lambda r: r[name], reverse=True)
            target_ranks = [i + 1 for i, row in enumerate(ordered) if row["is_acceptable_target"]]
            observable = bool(annotations[identifier].get("observable", bool(target_ranks)))
            rank_rows.append(
                {
                    "anonymous_id": identifier,
                    "target_observable": observable,
                    "best_target_rank": min(target_ranks) if target_ranks else None,
                }
            )
        metrics[name] = {
            "target": summary(target_values),
            "non_target": summary(other_values),
            "candidate_auc": auc_higher_is_target(target_values, other_values),
        }
        episode_rankings[name] = {
            "rank_1": sum(r["best_target_rank"] == 1 for r in rank_rows),
            "rank_at_most_2": sum(
                r["best_target_rank"] is not None and r["best_target_rank"] <= 2 for r in rank_rows
            ),
            "observable_episodes": sum(r["target_observable"] for r in rank_rows),
            "unobservable_episodes": sum(not r["target_observable"] for r in rank_rows),
            "rows": rank_rows,
        }

    result = {
        "schema_version": 1,
        "interpretation": "Change-point audit. Split role and whether parameters were frozen are determined by the supplied manifest and protocol.",
        "parameters": {
            "pre_frames_excluding_close": 15,
            "post_settle_frames": args.post_settle,
            "post_window_frames": args.post_window,
            "higher_score_means_larger_pre_to_post_motion_drop": True,
        },
        "counts": {
            "episodes": len(records),
            "observable_episodes": sum(bool(annotations[r["anonymous_id"]].get("observable", True)) for r in records),
            "unobservable_episodes": sum(not bool(annotations[r["anonymous_id"]].get("observable", True)) for r in records),
            "target_candidates": len(target),
            "non_target_candidates": len(other),
        },
        "metrics": metrics,
        "episode_rankings": episode_rankings,
        "candidates": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"counts": result["counts"], "metrics": metrics, "rankings": {k: {"rank_1": v["rank_1"], "rank_at_most_2": v["rank_at_most_2"]} for k, v in episode_rankings.items()}}, indent=2))


if __name__ == "__main__":
    main()
