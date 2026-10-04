#!/usr/bin/env python3
"""Search interpretable projected-gripper gates on consumed Wave 1."""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np


def quadratic_features(x: np.ndarray) -> np.ndarray:
    a, b, c = x.T
    return np.column_stack((np.ones(len(x)), a, b, c, a*a, b*b, c*c, a*b, a*c, b*c))


def points(values: list) -> np.ndarray:
    return np.asarray([[np.nan, np.nan] if value is None else value for value in values], dtype=float)


def cosine_synchrony(candidate: np.ndarray, gripper: np.ndarray, start: int, stop: int) -> float:
    a = np.diff(candidate[start:stop], axis=0)
    b = np.diff(gripper[start:stop], axis=0)
    valid = np.all(np.isfinite(a), axis=1) & np.all(np.isfinite(b), axis=1)
    valid &= (np.linalg.norm(a, axis=1) > 0.25) & (np.linalg.norm(b, axis=1) > 0.25)
    if valid.sum() < 3:
        return 0.0
    cosine = np.sum(a[valid] * b[valid], axis=1) / (
        np.linalg.norm(a[valid], axis=1) * np.linalg.norm(b[valid], axis=1) + 1e-9
    )
    return float(np.median(cosine))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--robot-state", type=Path, required=True)
    parser.add_argument("--projection", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    predictions = json.loads(args.predictions.read_text(encoding="utf-8"))["records"]
    features = {row["anonymous_id"]: row for row in json.loads(args.features.read_text(encoding="utf-8"))["records"]}
    states = {row["anonymous_id"]: row for row in json.loads(args.robot_state.read_text(encoding="utf-8"))["records"]}
    truth = {row["anonymous_id"]: set(row["acceptable_manipulated_candidate_ids"]) for row in json.loads(args.annotations.read_text(encoding="utf-8"))["records"]}
    projection = json.loads(args.projection.read_text(encoding="utf-8"))
    normalization = projection["coordinate_normalization"]
    mean, std = np.asarray(normalization["mean"]), np.asarray(normalization["std"])
    transform = np.asarray(projection["best_model"]["transform"])

    episode_data = []
    for prediction in predictions:
        identifier = prediction["anonymous_id"]
        xyz = np.asarray(states[identifier]["eef_position_xyz"], dtype=float)
        gripper = quadratic_features((xyz - mean) / std) @ transform
        candidates = {}
        for candidate_id, raw in prediction["centroids_xy"].items():
            track = points(raw)
            events = []
            for close in features[identifier]["close_frames"]:
                stop = min(len(track), close + 41)
                if stop - close < 12:
                    continue
                distance = np.linalg.norm(track[close:stop] - gripper[close:stop], axis=1)
                event = next(item for item in features[identifier]["candidates"][candidate_id]["events"] if item["close_frame"] == close)
                events.append({
                    "close": close,
                    "minimum_distance_px": float(np.nanmin(distance)) if np.isfinite(distance).any() else np.inf,
                    "median_distance_px": float(np.nanmedian(distance)) if np.isfinite(distance).any() else np.inf,
                    "displacement": event["post40_max_displacement"] or 0.0,
                    "background_residual": event["post40_residual_sum"],
                    "synchrony": cosine_synchrony(track, gripper, close, stop),
                })
            candidates[int(candidate_id)] = events
        episode_data.append((identifier, candidates))

    results = []
    for radius, sync_weight, residual_weight, minimum_motion in itertools.product(
        (20.0, 30.0, 40.0, 50.0, 65.0), (0.0, 0.25, 0.5, 1.0), (0.0, 0.1, 0.25), (0.005, 0.01, 0.02)
    ):
        correct = wrong = unknown = 0
        records = []
        for identifier, candidates in episode_data:
            scores = {}
            evidence = {}
            for candidate_id, events in candidates.items():
                eligible = [event for event in events if event["minimum_distance_px"] <= radius and event["displacement"] >= minimum_motion]
                if not eligible:
                    scores[candidate_id] = -np.inf
                    continue
                event_scores = [
                    event["displacement"] * (1 + sync_weight * max(event["synchrony"], 0.0))
                    + residual_weight * event["background_residual"]
                    for event in eligible
                ]
                best_index = int(np.argmax(event_scores))
                scores[candidate_id] = event_scores[best_index]
                evidence[candidate_id] = eligible[best_index]
            ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
            chosen = ranked[0][0] if ranked and np.isfinite(ranked[0][1]) else None
            status = "unknown" if chosen is None else ("correct" if chosen in truth[identifier] else "wrong")
            correct += status == "correct"
            wrong += status == "wrong"
            unknown += status == "unknown"
            records.append({"anonymous_id": identifier, "chosen": chosen, "status": status, "evidence": evidence.get(chosen)})
        results.append({
            "radius_px": radius, "synchrony_weight": sync_weight, "residual_weight": residual_weight,
            "minimum_motion": minimum_motion, "correct": correct, "wrong": wrong, "unknown": unknown,
            "records": records,
        })
    results.sort(key=lambda row: (-row["correct"], row["wrong"], row["radius_px"], row["synchrony_weight"]))
    args.output.write_text(json.dumps({"schema_version": 1, "records": results}, indent=2) + "\n", encoding="utf-8")
    for row in results[:15]:
        print({key: row[key] for key in ("radius_px", "synchrony_weight", "residual_weight", "minimum_motion", "correct", "wrong", "unknown")})


if __name__ == "__main__":
    main()
