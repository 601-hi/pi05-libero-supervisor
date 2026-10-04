"""Calibrate a persistent normal-gripper novelty alarm and evaluate it."""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np


def episode_groups(data: np.lib.npyio.NpzFile) -> list[np.ndarray]:
    groups: dict[str, list[int]] = collections.defaultdict(list)
    for i, episode in enumerate(data["episode_id"]):
        groups[str(episode)].append(i)
    return [
        np.asarray(sorted(indices, key=lambda i: int(data["action_index"][i])), dtype=np.int64)
        for indices in groups.values()
    ]


def persistent(raw: np.ndarray, required: int, window: int) -> np.ndarray:
    result = np.zeros(len(raw), dtype=bool)
    for i in range(len(raw)):
        result[i] = int(raw[max(0, i - window + 1) : i + 1].sum()) >= required
    return result


def normal_episode_false_alarms(data, threshold: float, required: int, window: int) -> int:
    score = -data["normal_logp"]
    return sum(bool(persistent(score[g] > threshold, required, window).any()) for g in episode_groups(data))


def calibrate(data, budget: int, required: int, window: int) -> float:
    if data["abnormal"].any():
        raise ValueError("calibration data must be normal-only")
    candidates = np.r_[np.nextafter(np.min(-data["normal_logp"]), -np.inf), np.unique(-data["normal_logp"])]
    valid = [
        value
        for value in candidates
        if normal_episode_false_alarms(data, float(value), required, window) <= budget
    ]
    if not valid:
        raise RuntimeError("no threshold satisfies the requested episode budget")
    return float(min(valid))


def evaluate(data, threshold: float, required: int, window: int) -> dict[str, object]:
    score = -data["normal_logp"]
    alarm = np.zeros(len(score), dtype=bool)
    details = []
    for indices in episode_groups(data):
        episode_alarm = persistent(score[indices] > threshold, required, window)
        alarm[indices] = episode_alarm
        active = data["abnormal"][indices].astype(bool)
        onset = next((j for j, value in enumerate(active) if value), None)
        first_alarm = next((j for j, value in enumerate(episode_alarm) if value), None)
        details.append(
            {
                "episode_id": str(data["episode_id"][indices[0]]),
                "has_active_fault": onset is not None,
                "any_alarm": bool(episode_alarm.any()),
                "detected_during_active": bool((episode_alarm & active).any()),
                "delay_steps": None if onset is None or first_alarm is None else first_alarm - onset,
                "pre_onset_alarm": bool(
                    onset is not None and first_alarm is not None and first_alarm < onset
                ),
            }
        )
    active = data["abnormal"].astype(bool)
    normal_details = [row for row in details if not row["has_active_fault"]]
    fault_details = [row for row in details if row["has_active_fault"]]
    return {
        "episodes": len(details),
        "normal_episodes": len(normal_details),
        "normal_episode_false_alarms": sum(row["any_alarm"] for row in normal_details),
        "abnormal_episodes": len(fault_details),
        "abnormal_episode_detections": sum(row["detected_during_active"] for row in fault_details),
        "pre_onset_alarm_episodes": sum(row["pre_onset_alarm"] for row in fault_details),
        "active_step_recall": float(alarm[active].mean()) if active.any() else None,
        "inactive_step_alarm_rate": float(alarm[~active].mean()) if (~active).any() else None,
        "details": details,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--normal-episode-budget", type=int, default=1)
    parser.add_argument("--persistence-required", type=int, default=2)
    parser.add_argument("--persistence-window", type=int, default=3)
    args = parser.parse_args()
    calibration = np.load(args.calibration, allow_pickle=False)
    evaluation = np.load(args.evaluation, allow_pickle=False)
    threshold = calibrate(
        calibration,
        args.normal_episode_budget,
        args.persistence_required,
        args.persistence_window,
    )
    report = {
        "version": 1,
        "score": "negative normal log likelihood of 2D gripper joint increment",
        "threshold": threshold,
        "normal_episode_budget": args.normal_episode_budget,
        "persistence_required": args.persistence_required,
        "persistence_window": args.persistence_window,
        "calibration_false_alarms": normal_episode_false_alarms(
            calibration, threshold, args.persistence_required, args.persistence_window
        ),
        "evaluation": evaluate(
            evaluation, threshold, args.persistence_required, args.persistence_window
        ),
    }
    text = json.dumps(report, ensure_ascii=False, indent=2)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
