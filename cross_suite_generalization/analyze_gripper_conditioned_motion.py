#!/usr/bin/env python3
"""Audit whether post-closure motion identifies the reviewed object on development data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def first_at_threshold(points: list, start: int, end: int, threshold: float, diagonal: float):
    if points[start] is None:
        return None
    origin = np.asarray(points[start], dtype=float)
    for index in range(start + 1, end):
        if points[index] is not None and np.linalg.norm(np.asarray(points[index]) - origin) / diagonal >= threshold:
            return index
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--review-grid", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-window", type=int, default=80)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    pilot = json.loads(args.pilot.read_text(encoding="utf-8"))
    review = json.loads(args.review_grid.read_text(encoding="utf-8"))
    expected = {row["goal_index"]: row["expected_manipulated_id_from_blind_review"]
                for row in review["configurations"][0]["episodes"]}
    first_by_goal = {}
    for job in manifest["jobs"]:
        first_by_goal.setdefault(job["goal_language"], job)

    episodes = []
    for episode in pilot["episodes"]:
        job = first_by_goal[episode["goal_language"]]
        rows = [json.loads(line) for line in Path(job["trace"]).read_text(encoding="utf-8").splitlines() if line]
        steps = sorted(
            [row for row in rows if row.get("event") == "step" and row.get("episode_idx") == job["episode_idx"]],
            key=lambda row: row["action_index"],
        )
        command = np.asarray([row["intended_action"][-1] for row in steps], dtype=float)
        aperture = np.asarray([
            abs(float(row["gripper_qpos_after"][0]) - float(row["gripper_qpos_after"][1]))
            for row in steps
        ])
        close_events = np.flatnonzero((command[1:] > 0) & (command[:-1] <= 0)) + 1
        candidates = {row["object_id"]: row["centroid_xy"] for row in episode["motion_ranked_candidates"]}
        diagonal = float(np.hypot(224, 224))
        events = []
        for start in close_events.tolist():
            next_open = next((idx for idx in range(start + 1, len(command)) if command[idx] <= 0), len(command))
            end = min(next_open, start + args.max_window, len(command))
            rankings = []
            for object_id, points in candidates.items():
                if start >= len(points) or points[start] is None:
                    displacement = 0.0
                else:
                    origin = np.asarray(points[start], dtype=float)
                    valid = [np.asarray(point, dtype=float) for point in points[start:end] if point is not None]
                    displacement = max((float(np.linalg.norm(point - origin) / diagonal) for point in valid), default=0.0)
                rankings.append({"object_id": object_id, "post_closure_max_displacement": displacement})
            rankings.sort(key=lambda row: row["post_closure_max_displacement"], reverse=True)
            expected_id = expected[episode["goal_index"]]
            expected_rank = next(index + 1 for index, row in enumerate(rankings) if row["object_id"] == expected_id)
            expected_points = candidates[expected_id]
            events.append({
                "closure_start": start,
                "window_end_exclusive": end,
                "aperture_at_start": float(aperture[start]),
                "minimum_aperture_in_window": float(aperture[start:end].min()),
                "expected_id": expected_id,
                "expected_rank": expected_rank,
                "expected_motion_onset_0p01": first_at_threshold(expected_points, start, end, 0.01, diagonal),
                "expected_motion_onset_0p03": first_at_threshold(expected_points, start, end, 0.03, diagonal),
                "rankings": rankings,
            })
        episodes.append({"goal_index": episode["goal_index"], "goal_language": episode["goal_language"], "events": events})
    result = {
        "schema_version": 1,
        "scope": "five development episodes; reviewed identity but no outcome used",
        "event_definition": "negative/nonpositive to positive gripper command",
        "episodes": episodes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    flat = [(row["goal_index"], event) for row in episodes for event in row["events"]]
    print(json.dumps({
        "episodes": len(episodes), "closure_events": len(flat),
        "expected_ranks": [{"goal": goal, "start": event["closure_start"], "rank": event["expected_rank"],
                            "onset_0p01": event["expected_motion_onset_0p01"],
                            "onset_0p03": event["expected_motion_onset_0p03"]} for goal, event in flat],
    }, indent=2))


if __name__ == "__main__":
    main()
