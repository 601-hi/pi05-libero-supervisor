#!/usr/bin/env python3
"""Evaluate interpretable wrist-contact candidate evidence on development data.

This is a candidate-ranking audit, not an outcome classifier.  The score uses
only close-frame geometry, pre/post-close motion change, and post-close track
visibility.  Episode outcome and task success are never inputs.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def distance_to_box(point: tuple[float, float], box: list[float]) -> float:
    x, y = point
    left, top, width, height = map(float, box)
    right, bottom = left + width, top + height
    dx = max(left - x, 0.0, x - right)
    dy = max(top - y, 0.0, y - bottom)
    return float(math.hypot(dx, dy))


def rank_fraction(values: dict[int, float], *, higher_is_better: bool = True) -> dict[int, float]:
    ordered = sorted(values, key=lambda key: values[key], reverse=higher_is_better)
    denominator = max(1, len(ordered) - 1)
    return {key: 1.0 - rank / denominator for rank, key in enumerate(ordered)}


def visibility(values: list, start: int = 5, stop: int = 20) -> float:
    selected = values[start:stop]
    return float(np.mean([value is not None for value in selected])) if selected else 0.0


def evaluate(rows: list[dict], annotations: dict, score_name: str) -> dict:
    ranks = []
    for row in rows:
        acceptable = set(annotations[row["anonymous_id"]]["acceptable_candidate_ids"])
        ordered = sorted(row["candidates"], key=lambda candidate: candidate[score_name], reverse=True)
        target_ranks = [index + 1 for index, candidate in enumerate(ordered) if candidate["candidate_id"] in acceptable]
        observable = bool(annotations[row["anonymous_id"]].get("observable", bool(acceptable)))
        ranks.append({
            "anonymous_id": row["anonymous_id"],
            "observable": observable,
            "best_target_rank": min(target_ranks) if target_ranks else None,
        })
    observable = [row for row in ranks if row["observable"]]
    return {
        "observable_episodes": len(observable),
        "rank_1": sum(row["best_target_rank"] == 1 for row in observable),
        "rank_at_most_2": sum(row["best_target_rank"] is not None and row["best_target_rank"] <= 2 for row in observable),
        "mean_reciprocal_rank": float(np.mean([
            0.0 if row["best_target_rank"] is None else 1.0 / row["best_target_rank"] for row in observable
        ])) if observable else float("nan"),
        "rows": ranks,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--changepoint", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pinch-x", type=float, default=112.0)
    parser.add_argument("--pinch-y", type=float, default=184.0)
    parser.add_argument("--image-size", type=float, default=224.0)
    args = parser.parse_args()
    predictions = json.loads(args.predictions.read_text(encoding="utf-8"))["records"]
    change_rows = json.loads(args.changepoint.read_text(encoding="utf-8"))["candidates"]
    change = {(row["anonymous_id"], int(row["candidate_id"])): row for row in change_rows}
    annotations = json.loads(args.annotations.read_text(encoding="utf-8"))["annotations"]

    episodes = []
    for record in predictions:
        identifier = record["anonymous_id"]
        raw_change = {
            int(key): float(change[(identifier, int(key))]["combined_change_score"])
            for key in record["post_centroids_xy"]
        }
        change_rank = rank_fraction(raw_change)
        candidates = []
        for candidate_index, box in enumerate(record["boxes_xywh"], 1):
            key = str(candidate_index)
            proximity = math.exp(-distance_to_box((args.pinch_x, args.pinch_y), box) / 32.0)
            visible = visibility(record["post_centroids_xy"][key])
            left, top, width, height = map(float, box)
            robot_band = float(width >= 0.80 * args.image_size and top >= 0.72 * args.image_size)
            near_full_frame = float(width * height >= 0.70 * args.image_size * args.image_size)
            score = (
                0.55 * change_rank[candidate_index]
                + 0.25 * proximity
                + 0.20 * visible
                - 0.35 * robot_band
                - 0.20 * near_full_frame
            )
            candidates.append({
                "candidate_id": candidate_index,
                "combined_change_score": raw_change[candidate_index],
                "change_rank_fraction": change_rank[candidate_index],
                "pinch_box_distance_px": distance_to_box((args.pinch_x, args.pinch_y), box),
                "pinch_proximity": proximity,
                "postclose_visibility": visible,
                "robot_bottom_band_penalty": robot_band,
                "near_full_frame_penalty": near_full_frame,
                "physical_fusion_score": score,
            })
        episodes.append({"anonymous_id": identifier, "candidates": candidates})

    metrics = {
        "changepoint_only": evaluate(episodes, annotations, "combined_change_score"),
        "pinch_proximity_only": evaluate(episodes, annotations, "pinch_proximity"),
        "visibility_only": evaluate(episodes, annotations, "postclose_visibility"),
        "physical_fusion": evaluate(episodes, annotations, "physical_fusion_score"),
    }
    output = {
        "schema_version": 1,
        "development_only": True,
        "forbidden_inputs": ["episode outcome", "reward", "task success"],
        "geometry_prior": {"pinch_center_xy": [args.pinch_x, args.pinch_y], "image_size": args.image_size},
        "score_definition": "0.55*within_episode_change_rank + 0.25*box_pinch_proximity + 0.20*post_visibility - 0.35*robot_bottom_band - 0.20*near_full_frame",
        "metrics": metrics,
        "episodes": episodes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({name: {k: v[k] for k in ("observable_episodes", "rank_1", "rank_at_most_2", "mean_reciprocal_rank")} for name, v in metrics.items()}, indent=2))


if __name__ == "__main__":
    main()
