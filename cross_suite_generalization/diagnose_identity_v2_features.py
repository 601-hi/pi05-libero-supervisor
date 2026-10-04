#!/usr/bin/env python3
"""Extract outcome-free causal/background features for consumed Wave 1 candidates."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vla_supervisor.background_motion import estimate_background_motion


def load_trace_rows(trace_root: Path, private: dict, action_indices: np.ndarray) -> list[dict]:
    pattern = f"{private['suite']}_task{private['task_id']}_*jsonl"
    matches = list(trace_root.glob(pattern))
    if len(matches) != 1:
        raise ValueError(f"expected one trace for {pattern}, found {matches}")
    episode_rows = {}
    for line in matches[0].open(encoding="utf-8"):
        row = json.loads(line)
        if row.get("event") == "step" and row.get("episode_idx") == private["episode_idx"]:
            episode_rows[int(row["action_index"])] = row
    missing = [int(index) for index in action_indices if int(index) not in episode_rows]
    if missing:
        raise ValueError(f"missing action indices for {private['anonymous_id']}: {missing[:5]}")
    return [episode_rows[int(index)] for index in action_indices]


def closing_events(commands: np.ndarray, close_threshold: float = 0.5, open_threshold: float = -0.5) -> list[int]:
    events, armed = [], True
    for index, command in enumerate(commands):
        if command <= open_threshold:
            armed = True
        elif command >= close_threshold and armed:
            events.append(index)
            armed = False
    return events


def project(point: np.ndarray, homography: np.ndarray) -> np.ndarray:
    return cv2.perspectiveTransform(point.reshape(1, 1, 2).astype(np.float32), homography)[0, 0]


def candidate_features(
    points: list,
    background: list,
    closes: list[int],
    diagonal: float,
) -> dict:
    array = np.asarray([[np.nan, np.nan] if point is None else point for point in points], dtype=float)
    residual = np.full(len(array), np.nan)
    raw_step = np.full(len(array), np.nan)
    for index in range(1, len(array)):
        if not np.all(np.isfinite(array[index - 1 : index + 1])):
            continue
        raw_step[index] = np.linalg.norm(array[index] - array[index - 1]) / diagonal
        estimate = background[index]
        if estimate.valid and estimate.homography is not None:
            residual[index] = np.linalg.norm(array[index] - project(array[index - 1], estimate.homography)) / diagonal

    event_features = []
    for close in closes:
        pre_start = max(0, close - 20)
        post_stop = min(len(array), close + 41)
        origin = array[close] if close < len(array) else np.asarray([np.nan, np.nan])
        post_displacement = np.linalg.norm(array[close:post_stop] - origin, axis=1) / diagonal
        pre_residual = residual[pre_start:close]
        post_residual = residual[close + 1 : post_stop]
        onset = next((offset for offset, value in enumerate(post_displacement) if np.isfinite(value) and value >= 0.01), None)
        event_features.append({
            "close_frame": close,
            "post40_max_displacement": float(np.nanmax(post_displacement)) if np.isfinite(post_displacement).any() else None,
            "post40_displacement_onset_001": onset,
            "pre20_background_residual_median": float(np.nanmedian(pre_residual)) if np.isfinite(pre_residual).any() else None,
            "post40_background_residual_median": float(np.nanmedian(post_residual)) if np.isfinite(post_residual).any() else None,
            "post40_background_residual_p90": float(np.nanpercentile(post_residual, 90)) if np.isfinite(post_residual).any() else None,
            "post40_residual_sum": float(np.nansum(post_residual)),
        })

    finite = np.all(np.isfinite(array), axis=1)
    origin_index = int(np.flatnonzero(finite)[0]) if finite.any() else None
    net = None
    if origin_index is not None:
        net = float(np.nanmax(np.linalg.norm(array - array[origin_index], axis=1) / diagonal))
    return {
        "maximum_net_displacement": net,
        "raw_step_median": float(np.nanmedian(raw_step)) if np.isfinite(raw_step).any() else None,
        "background_residual_median": float(np.nanmedian(residual)) if np.isfinite(residual).any() else None,
        "background_residual_p90": float(np.nanpercentile(residual, 90)) if np.isfinite(residual).any() else None,
        "background_valid_fraction": float(np.mean(np.isfinite(residual[1:]))) if len(residual) > 1 else 0.0,
        "background_residual_sequence_normalized": [
            None if not np.isfinite(value) else float(value) for value in residual
        ],
        "events": event_features,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--public", type=Path, required=True)
    parser.add_argument("--private", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--trace-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    public = json.loads(args.public.read_text(encoding="utf-8"))
    private = json.loads(args.private.read_text(encoding="utf-8"))
    predictions = json.loads(args.predictions.read_text(encoding="utf-8"))["records"]
    annotations = (
        json.loads(args.annotations.read_text(encoding="utf-8"))["records"]
        if args.annotations is not None else []
    )
    public_by_id = {row["anonymous_id"]: row for row in public["records"]}
    private_by_id = {row["anonymous_id"]: row for row in private["records"]}
    annotation_by_id = {row["anonymous_id"]: row for row in annotations}
    output = []

    for episode_index, prediction in enumerate(predictions, start=1):
        anonymous_id = prediction["anonymous_id"]
        private_row = private_by_id[anonymous_id]
        with np.load(private_row["original_sidecar"], mmap_mode="r") as data:
            frames = np.asarray(data["agent_images"], dtype=np.uint8)
            action_indices = np.asarray(data["action_indices"], dtype=int)
        steps = load_trace_rows(args.trace_root, private_row, action_indices)
        commands = np.asarray([row["intended_action"][6] for row in steps], dtype=float)
        closes = closing_events(commands)
        background = [None]
        for frame_index in range(1, len(frames)):
            background.append(estimate_background_motion(frames[frame_index - 1], frames[frame_index]))
        diagonal = float(np.hypot(*frames.shape[1:3]))
        candidates = {
            str(candidate_id): candidate_features(points, background, closes, diagonal)
            for candidate_id, points in prediction["centroids_xy"].items()
        }
        output.append({
            "anonymous_id": anonymous_id,
            "goal_language": public_by_id[anonymous_id]["goal_language"],
            "close_frames": closes,
            "locked_candidate_id_v1": prediction["locked_candidate_id"],
            "acceptable_candidate_ids": (
                annotation_by_id[anonymous_id]["acceptable_manipulated_candidate_ids"]
                if anonymous_id in annotation_by_id else None
            ),
            "background_valid_fraction": float(np.mean([item.valid for item in background[1:]])),
            "background_confidence_sequence": [
                0.0 if item is None else float(item.confidence) for item in background
            ],
            "background_valid_sequence": [
                False if item is None else bool(item.valid) for item in background
            ],
            "candidates": candidates,
        })
        print(json.dumps({"episode": episode_index, "anonymous_id": anonymous_id, "frames": len(frames), "closes": closes}), flush=True)

    args.output.write_text(json.dumps({"schema_version": 1, "records": output}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
