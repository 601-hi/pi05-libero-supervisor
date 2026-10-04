#!/usr/bin/env python3
"""Build outcome-blind fixed-view motion evidence around gripper-close events.

The script deliberately consumes only the public contact manifest and anonymous
candidate tracks.  It must not read task language, rewards, or episode outcome.
Its output is supporting evidence for physical annotation, not a ground-truth
label and not a target-object prediction.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _point(value: object) -> np.ndarray | None:
    if not isinstance(value, list) or len(value) != 2:
        return None
    point = np.asarray(value, dtype=float)
    return point if np.all(np.isfinite(point)) else None


def _step_lengths(points: list[np.ndarray | None], start: int, end: int) -> list[float]:
    lengths: list[float] = []
    for first, second in zip(points[max(0, start):end - 1], points[max(0, start) + 1:end]):
        if first is not None and second is not None:
            lengths.append(float(np.linalg.norm(second - first)))
    return lengths


def summarize_candidate(sequence: list, close: int, pre: int, post: int) -> dict:
    points = [_point(value) for value in sequence]
    start = max(0, close - pre)
    end = min(len(points), close + post + 1)
    pre_steps = _step_lengths(points, start, close + 1)
    pre_noise = float(np.median(pre_steps)) if pre_steps else 0.0
    # Three pixels prevents subpixel tracker jitter from being called motion;
    # the noise-adaptive term handles unusually unstable candidates.
    threshold = float(max(3.0, 4.0 * pre_noise))
    origin = points[close] if close < len(points) else None
    valid_post = [(index, points[index]) for index in range(close, end) if points[index] is not None]
    if origin is None or not valid_post:
        return {
            "preclose_median_step_px": pre_noise,
            "adaptive_motion_threshold_px": threshold,
            "postclose_visible_fraction": 0.0,
            "postclose_net_displacement_px": None,
            "postclose_max_displacement_px": None,
            "motion_onset_frame": None,
            "sustained_motion": False,
        }
    distances = [(index, float(np.linalg.norm(point - origin))) for index, point in valid_post]
    above = [index for index, distance in distances if distance >= threshold]
    tail = [distance for _, distance in distances[-min(5, len(distances)):]]
    return {
        "preclose_median_step_px": pre_noise,
        "adaptive_motion_threshold_px": threshold,
        "postclose_visible_fraction": len(valid_post) / max(1, end - close),
        "postclose_net_displacement_px": distances[-1][1],
        "postclose_max_displacement_px": max(distance for _, distance in distances),
        "motion_onset_frame": above[0] if above else None,
        "sustained_motion": bool(tail and np.median(tail) >= threshold),
    }


def co_motion_groups(
    sequences: dict[str, list], close: int, post: int, sustained_ids: set[str],
    *, difference_threshold_px: float = 4.0, minimum_overlap: int = 5,
) -> list[list[str]]:
    """Group candidates with similar post-close *relative* 2-D trajectories.

    This collapses duplicate masks and also represents rigid co-motion between a
    grasped object and robot pixels.  A group is evidence of one motion mode,
    not proof that all members are the same semantic object.
    """
    identifiers = sorted(sustained_ids)
    parent = {identifier: identifier for identifier in identifiers}

    def find(value: str) -> str:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(first: str, second: str) -> None:
        first_root, second_root = find(first), find(second)
        if first_root != second_root:
            parent[second_root] = first_root

    prepared = {key: [_point(value) for value in sequences[key]] for key in identifiers}
    for first_index, first_id in enumerate(identifiers):
        first = prepared[first_id]
        if close >= len(first) or first[close] is None:
            continue
        for second_id in identifiers[first_index + 1:]:
            second = prepared[second_id]
            if close >= len(second) or second[close] is None:
                continue
            differences = []
            end = min(len(first), len(second), close + post + 1)
            for frame in range(close, end):
                if first[frame] is not None and second[frame] is not None:
                    first_delta = first[frame] - first[close]
                    second_delta = second[frame] - second[close]
                    differences.append(float(np.linalg.norm(first_delta - second_delta)))
            if len(differences) >= minimum_overlap and np.median(differences) <= difference_threshold_px:
                union(first_id, second_id)
    grouped: dict[str, list[str]] = {}
    for identifier in identifiers:
        grouped.setdefault(find(identifier), []).append(identifier)
    return sorted(grouped.values(), key=lambda group: (-len(group), group))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pre-frames", type=int, default=10)
    parser.add_argument("--post-frames", type=int, default=40)
    args = parser.parse_args()
    if args.pre_frames < 2 or args.post_frames < 2:
        raise ValueError("pre-frames and post-frames must both be at least 2")

    public = json.loads(args.manifest.read_text(encoding="utf-8"))["records"]
    tracks: dict[str, dict] = {}
    for path in args.predictions:
        for record in json.loads(path.read_text(encoding="utf-8"))["records"]:
            identifier = record["anonymous_id"]
            if identifier in tracks:
                raise ValueError(f"duplicate anonymous id: {identifier}")
            tracks[identifier] = record

    records = []
    missing = []
    for episode in public:
        identifier = episode["anonymous_id"]
        prediction = tracks.get(identifier)
        if prediction is None:
            missing.append(identifier)
            continue
        events = []
        for close in episode["close_frames"]:
            candidates = []
            for candidate_id, sequence in prediction["centroids_xy"].items():
                candidates.append({
                    "candidate_id": candidate_id,
                    **summarize_candidate(sequence, int(close), args.pre_frames, args.post_frames),
                })
            candidates.sort(
                key=lambda row: (
                    row["sustained_motion"],
                    row["postclose_max_displacement_px"] or -1.0,
                    row["postclose_visible_fraction"],
                ),
                reverse=True,
            )
            for rank, row in enumerate(candidates, 1):
                row["motion_rank"] = rank
            sustained_ids = {row["candidate_id"] for row in candidates if row["sustained_motion"]}
            events.append({
                "close_frame": int(close),
                "locked_candidate_id_from_tracker": str(prediction["locked_candidate_id"]),
                "co_motion_groups": co_motion_groups(
                    prediction["centroids_xy"], int(close), args.post_frames, sustained_ids
                ),
                "candidates": candidates,
            })
        records.append({"anonymous_id": identifier, "events": events})

    result = {
        "schema_version": 1,
        "scope": "outcome-blind fixed-view candidate motion evidence; not labels",
        "forbidden_inputs": ["task language", "episode outcome", "reward", "semantic class"],
        "pre_frames": args.pre_frames,
        "post_frames": args.post_frames,
        "episodes": len(records),
        "events": sum(len(record["events"]) for record in records),
        "missing_anonymous_ids": missing,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("episodes", "events", "missing_anonymous_ids")}, indent=2))


if __name__ == "__main__":
    main()
