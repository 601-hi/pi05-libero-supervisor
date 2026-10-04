"""Select multiple label-blind visual events and matched controls per episode.

The independent evaluation unit remains an episode (and is summarized by task/seed).
Windows from one episode are deliberately *not* treated as independent samples.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def candidate_windows(steps: list[dict], length: int, minimum_target_path: float) -> list[dict]:
    result = []
    denominator = max(1, len(steps) - length)
    for start in range(len(steps) - length + 1):
        window = steps[start : start + length]
        target = np.asarray([row["intended_target_translation"] for row in window], dtype=float)
        actual = np.asarray([row["actual_translation"] for row in window], dtype=float)
        target_path = float(np.linalg.norm(target, axis=1).sum())
        if target_path < minimum_target_path:
            continue
        actual_path = float(np.linalg.norm(actual, axis=1).sum())
        result.append({
            "start_action_index": int(window[0]["action_index"]),
            "end_action_index": int(window[-1]["action_index"]),
            "phase": float(start / denominator),
            "target_path_m": target_path,
            "actual_path_m": actual_path,
            "response_ratio": actual_path / target_path,
        })
    return result


def overlaps(a: dict, b: dict, minimum_gap: int) -> bool:
    return not (
        a["end_action_index"] + minimum_gap < b["start_action_index"]
        or b["end_action_index"] + minimum_gap < a["start_action_index"]
    )


def separated_extremes(windows: list[dict], count: int, minimum_gap: int, reverse: bool) -> list[dict]:
    ordered = sorted(windows, key=lambda row: row["response_ratio"], reverse=reverse)
    selected: list[dict] = []
    for row in ordered:
        if all(not overlaps(row, prior, minimum_gap) for prior in selected):
            selected.append(row)
            if len(selected) == count:
                break
    return sorted(selected, key=lambda row: row["start_action_index"])


def matched_controls(windows: list[dict], events: list[dict], minimum_gap: int) -> list[dict]:
    """Match controls on episode phase and commanded path, without outcome labels."""
    if not windows:
        return []
    response_median = float(np.median([row["response_ratio"] for row in windows]))
    pool = [row for row in windows if row["response_ratio"] >= response_median]
    chosen: list[dict] = []
    for event in events:
        eligible = [row for row in pool if all(
            not overlaps(row, blocked, minimum_gap) for blocked in events + chosen
        )]
        if not eligible:
            continue
        def distance(row: dict) -> float:
            command_scale = max(event["target_path_m"], 1e-6)
            return abs(row["phase"] - event["phase"]) + abs(
                row["target_path_m"] - event["target_path_m"]
            ) / command_scale
        selected = min(eligible, key=distance)
        selected = dict(selected)
        selected["matched_event_start_action_index"] = event["start_action_index"]
        selected["match_distance"] = distance(selected)
        chosen.append(selected)
    return sorted(chosen, key=lambda row: row["start_action_index"])


def load_episode_steps(trace_path: Path, episode_idx: int) -> list[dict]:
    return [
        row for line in trace_path.open(encoding="utf-8")
        if (row := json.loads(line)).get("event") == "step"
        and int(row["episode_idx"]) == episode_idx
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--events-per-episode", type=int, default=2)
    parser.add_argument("--window-length", type=int, default=5)
    parser.add_argument("--minimum-gap", type=int, default=10)
    parser.add_argument("--minimum-target-path", type=float, default=0.02)
    args = parser.parse_args()

    source = json.loads(args.input_manifest.read_text(encoding="utf-8"))
    episodes = []
    for episode in source["episodes"]:
        row = dict(episode)
        steps = load_episode_steps(Path(row["trace_path"]), int(row["episode_idx"]))
        windows = candidate_windows(steps, args.window_length, args.minimum_target_path)
        events = separated_extremes(windows, args.events_per_episode, args.minimum_gap, reverse=False)
        controls = matched_controls(windows, events, args.minimum_gap)
        row["label_blind_low_response_events"] = events
        row["label_blind_matched_controls"] = controls
        episodes.append(row)

    result = {
        "schema_version": 2,
        "source_manifest": str(args.input_manifest.resolve()),
        "independence_warning": (
            "Windows within one episode are correlated. Evaluation confidence intervals must use "
            "episodes, tasks, and seeds as grouped units, never raw windows."
        ),
        "selection_firewall": [
            "event selection uses intended and actual translation only",
            "success, reward, language, and object identity are not used",
            "events are temporally separated",
            "controls are matched on episode phase and commanded translation path",
        ],
        "parameters": {
            "events_per_episode": args.events_per_episode,
            "window_length": args.window_length,
            "minimum_gap": args.minimum_gap,
            "minimum_target_path": args.minimum_target_path,
        },
        "counts": {
            "episodes": len(episodes),
            "events": sum(len(row["label_blind_low_response_events"]) for row in episodes),
            "controls": sum(len(row["label_blind_matched_controls"]) for row in episodes),
        },
        "episodes": episodes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result["counts"], ensure_ascii=False))


if __name__ == "__main__":
    main()
