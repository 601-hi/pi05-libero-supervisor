"""Pilot-only grid analysis for causal manipulated-object identity locking."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def displacement_series(candidate: dict, diagonal: float) -> list[float]:
    points = candidate["centroid_xy"]
    origin = next(point for point in points if point is not None)
    maximum = 0.0
    result = []
    for point in points:
        if point is not None:
            maximum = max(maximum, math.dist(point, origin) / diagonal)
        result.append(maximum)
    return result


def causal_lock(series: dict[int, list[float]], minimum: float, margin: float, persistence: int):
    streak_id = None
    streak = 0
    for frame in range(len(next(iter(series.values())))):
        ranked = sorted(((values[frame], object_id) for object_id, values in series.items()), reverse=True)
        qualifies = ranked[0][0] >= minimum and ranked[0][0] - ranked[1][0] >= margin
        candidate_id = ranked[0][1] if qualifies else None
        if candidate_id is not None and candidate_id == streak_id:
            streak += 1
        elif candidate_id is not None:
            streak_id, streak = candidate_id, 1
        else:
            streak_id, streak = None, 0
        if streak >= persistence:
            return streak_id, frame
    return None, None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.input.read_text(encoding="utf-8"))
    diagonal = math.hypot(224, 224)
    configurations = []
    for minimum in (0.03, 0.05, 0.07, 0.10):
        for margin in (0.02, 0.04, 0.06):
            for persistence in (2, 3, 5):
                episodes = []
                for episode in data["episodes"]:
                    series = {
                        int(candidate["object_id"]): displacement_series(candidate, diagonal)
                        for candidate in episode["motion_ranked_candidates"]
                    }
                    expected_id = int(episode["motion_ranked_candidates"][0]["object_id"])
                    selected_id, frame = causal_lock(series, minimum, margin, persistence)
                    episodes.append({
                        "goal_index": episode["goal_index"],
                        "expected_manipulated_id_from_blind_review": expected_id,
                        "selected_id": selected_id,
                        "lock_frame": frame,
                        "matches_review": selected_id == expected_id,
                    })
                locked = [row for row in episodes if row["selected_id"] is not None]
                configurations.append({
                    "minimum_motion": minimum,
                    "minimum_margin": margin,
                    "persistence": persistence,
                    "correct_locks": sum(row["matches_review"] for row in episodes),
                    "locks": len(locked),
                    "mean_lock_frame": (
                        sum(row["lock_frame"] for row in locked) / len(locked) if locked else None
                    ),
                    "episodes": episodes,
                })
    configurations.sort(
        key=lambda row: (-row["correct_locks"], row["locks"] - row["correct_locks"], row["mean_lock_frame"] or 1e9)
    )
    result = {
        "schema_version": 1,
        "scope": "pilot calibration only; expected ids were visually reviewed without outcome labels",
        "num_episodes": len(data["episodes"]),
        "configurations": configurations,
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(configurations[:10], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
