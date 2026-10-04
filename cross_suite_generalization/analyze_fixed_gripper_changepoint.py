#!/usr/bin/env python3
"""Audit fixed-view candidate/gripper change points around closure."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vla_supervisor.gripper_projection import GripperPixelProjector


def points(values: list) -> np.ndarray:
    return np.asarray([[np.nan, np.nan] if value is None else value for value in values], dtype=float)


def median_speed(track: np.ndarray, start: int, stop: int) -> float:
    selected = track[max(0, start):min(len(track), stop)]
    if len(selected) < 2:
        return float("nan")
    valid = np.all(np.isfinite(selected[:-1]), axis=1) & np.all(np.isfinite(selected[1:]), axis=1)
    speed = np.linalg.norm(np.diff(selected, axis=0), axis=1)[valid]
    return float(np.median(speed)) if len(speed) else float("nan")


def log_ratio(before: float, after: float, epsilon: float = .1) -> float:
    if not np.isfinite(before) or not np.isfinite(after):
        return float("nan")
    return float(np.log((before + epsilon) / (after + epsilon)))


def auc(target: list[float], other: list[float]) -> float:
    comparisons = [(left > right) + .5 * (left == right) for left in target for right in other if np.isfinite(left) and np.isfinite(right)]
    return float(np.mean(comparisons)) if comparisons else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--robot-state", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--projection", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pre-frames", type=int, default=10)
    parser.add_argument("--post-settle", type=int, default=3)
    parser.add_argument("--post-frames", type=int, default=15)
    args = parser.parse_args()
    predictions = json.loads(args.predictions.read_text(encoding="utf-8"))["records"]
    states = {row["anonymous_id"]: row for row in json.loads(args.robot_state.read_text(encoding="utf-8"))["records"]}
    features = {row["anonymous_id"]: row for row in json.loads(args.features.read_text(encoding="utf-8"))["records"]}
    truth = {row["anonymous_id"]: set(row["acceptable_target_mask_ids"]) for row in json.loads(args.annotations.read_text(encoding="utf-8"))["records"]}
    projector = GripperPixelProjector.from_fit_file(args.projection)
    records = []
    for prediction in predictions:
        identifier = prediction["anonymous_id"]
        gripper = np.asarray([projector.project(xyz).center_xy for xyz in states[identifier]["eef_position_xyz"]])
        candidates = {}
        for candidate_id, sequence in prediction["centroids_xy"].items():
            track = points(sequence)
            relative = track - gripper[:len(track)]
            events = []
            for close in features[identifier]["close_frames"]:
                pre_relative = median_speed(relative, close - args.pre_frames, close)
                post_relative = median_speed(relative, close + args.post_settle, close + args.post_settle + args.post_frames)
                pre_global = median_speed(track, close - args.pre_frames, close)
                post_global = median_speed(track, close + args.post_settle, close + args.post_settle + args.post_frames)
                events.append({
                    "close_frame": int(close),
                    "pre_relative_speed_px": pre_relative,
                    "post_relative_speed_px": post_relative,
                    "relative_speed_log_drop": log_ratio(pre_relative, post_relative),
                    "global_speed_log_rise": -log_ratio(pre_global, post_global),
                })
            valid = [event for event in events if np.isfinite(event["relative_speed_log_drop"]) and np.isfinite(event["global_speed_log_rise"])]
            candidates[candidate_id] = {
                "is_acceptable_target": int(candidate_id) in truth[identifier],
                "events": events,
                "best_relative_speed_log_drop": max((event["relative_speed_log_drop"] for event in valid), default=float("nan")),
                "best_combined_change": max((event["relative_speed_log_drop"] + event["global_speed_log_rise"] for event in valid), default=float("nan")),
            }
        records.append({"anonymous_id": identifier, "candidates": candidates})

    rows = [(int(cid), candidate) for record in records for cid, candidate in record["candidates"].items()]
    metrics = {}
    for score_name in ("best_relative_speed_log_drop", "best_combined_change"):
        target = [row[score_name] for _, row in rows if row["is_acceptable_target"]]
        other = [row[score_name] for _, row in rows if not row["is_acceptable_target"]]
        top1 = top2 = observable = 0
        for record in records:
            acceptable = [int(cid) for cid, value in record["candidates"].items() if value["is_acceptable_target"]]
            if not acceptable:
                continue
            observable += 1
            ordered = sorted(record["candidates"], key=lambda cid: (np.nan_to_num(record["candidates"][cid][score_name], nan=-1e9)), reverse=True)
            ranks = [ordered.index(str(candidate)) + 1 for candidate in acceptable]
            top1 += min(ranks) == 1
            top2 += min(ranks) <= 2
        metrics[score_name] = {"candidate_auc": auc(target, other), "observable_episodes": observable, "top1": top1, "top2": top2}
    output = {
        "schema_version": 1,
        "interpretation": "fixed-view candidate/gripper closure change point; development audit",
        "projection_uncertainty_p90_px": projector.uncertainty_p90_px,
        "parameters": {"pre_frames": args.pre_frames, "post_settle": args.post_settle, "post_frames": args.post_frames},
        "metrics": metrics,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
