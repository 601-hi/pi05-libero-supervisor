#!/usr/bin/env python3
"""Explore gripper-proxy synchrony on consumed Wave 1 candidate tracks."""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np


def array(points: list) -> np.ndarray:
    return np.asarray([[np.nan, np.nan] if point is None else point for point in points], dtype=float)


def velocity(points: np.ndarray) -> np.ndarray:
    result = np.diff(points, axis=0, prepend=points[:1])
    result[~np.all(np.isfinite(result), axis=1)] = np.nan
    return result


def path(velocities: np.ndarray, start: int, stop: int, diagonal: float) -> float:
    speed = np.linalg.norm(velocities[max(0, start) : min(stop, len(velocities))], axis=1)
    return float(np.nansum(speed) / diagonal)


def synchrony(first: np.ndarray, second: np.ndarray, start: int, stop: int) -> tuple[float, int]:
    a, b = first[start:stop], second[start:stop]
    valid = np.all(np.isfinite(a), axis=1) & np.all(np.isfinite(b), axis=1)
    valid &= (np.linalg.norm(a, axis=1) > 0.25) & (np.linalg.norm(b, axis=1) > 0.25)
    if valid.sum() < 3:
        return 0.0, int(valid.sum())
    cosine = np.sum(a[valid] * b[valid], axis=1) / (
        np.linalg.norm(a[valid], axis=1) * np.linalg.norm(b[valid], axis=1) + 1e-9
    )
    return float(np.median(cosine)), int(valid.sum())


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
    diagonal = float(np.hypot(224, 224))
    settings = itertools.product((1, 2, 3), (0.0, 0.25, 0.5, 1.0), (0.0, 0.01, 0.02))
    searches = []
    for proxy_count, synchrony_weight, minimum_score in settings:
        correct = unknown = 0
        records = []
        for prediction in predictions:
            episode = features[prediction["anonymous_id"]]
            tracks = {int(key): array(value) for key, value in prediction["centroids_xy"].items()}
            velocities = {key: velocity(value) for key, value in tracks.items()}
            candidate_scores = {key: -np.inf for key in tracks}
            candidate_evidence = {}
            for close in episode["close_frames"]:
                if len(next(iter(tracks.values()))) - close < 12:
                    continue
                pre_paths = sorted(
                    ((path(value, max(0, close - 30), close, diagonal), key) for key, value in velocities.items()),
                    reverse=True,
                )
                proxies = [key for _, key in pre_paths[:proxy_count]]
                for key, candidate_velocity in velocities.items():
                    if key in proxies:
                        continue
                    displacement = max(
                        (item["post40_max_displacement"] or 0.0)
                        for item in episode["candidates"][str(key)]["events"]
                        if item["close_frame"] == close
                    )
                    sync_values = [synchrony(candidate_velocity, velocities[proxy], close, min(close + 41, len(candidate_velocity)))[0] for proxy in proxies]
                    sync = max(sync_values, default=0.0)
                    score = displacement * (1.0 + synchrony_weight * max(sync, 0.0))
                    if score > candidate_scores[key]:
                        candidate_scores[key] = score
                        candidate_evidence[key] = {"close": close, "proxies": proxies, "displacement": displacement, "synchrony": sync}
            ranked = sorted(candidate_scores.items(), key=lambda item: item[1], reverse=True)
            chosen = ranked[0][0] if ranked and ranked[0][1] >= minimum_score else None
            correct += chosen in truth[prediction["anonymous_id"]]
            unknown += chosen is None
            records.append({"anonymous_id": prediction["anonymous_id"], "chosen": chosen, "score": ranked[0][1], "evidence": candidate_evidence.get(chosen)})
        searches.append({
            "proxy_count": proxy_count,
            "synchrony_weight": synchrony_weight,
            "minimum_score": minimum_score,
            "correct": correct,
            "unknown": unknown,
            "records": records,
        })
    searches.sort(key=lambda row: (-row["correct"], row["unknown"], row["proxy_count"], row["synchrony_weight"]))
    args.output.write_text(json.dumps({"schema_version": 1, "records": searches}, indent=2) + "\n", encoding="utf-8")
    for result in searches[:12]:
        print({key: result[key] for key in ("proxy_count", "synchrony_weight", "minimum_score", "correct", "unknown")})


if __name__ == "__main__":
    main()
