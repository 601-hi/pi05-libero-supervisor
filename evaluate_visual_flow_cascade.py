#!/usr/bin/env python3
"""Calibration-only evaluation of optical-flow rescue inside two-expert ambiguity."""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np


def classify_states(scores: np.lib.npyio.NpzFile, config: dict, task_ids: np.ndarray) -> np.ndarray:
    result = []
    for i, task_id in enumerate(task_ids):
        current = {**config, **config.get("by_task", {}).get(str(int(task_id)), {})}
        normal = float(scores["normal_logp"][i])
        abnormal = float(scores["abnormal_logp"][i])
        ratio = abnormal - normal
        normal_high = normal >= current["normal_absolute_floor"]
        abnormal_high = abnormal >= current["abnormal_absolute_floor"]
        if abnormal_high:
            state = "known_abnormal" if ratio >= current["ratio_abnormal_threshold"] else "ambiguous_overlap"
        elif normal_high:
            state = "normal"
        else:
            state = "unknown_abnormal"
        result.append(state)
    return np.asarray(result)


def persistent(points: np.ndarray, required: int = 2, window: int = 3) -> np.ndarray:
    alarm = np.zeros(len(points), dtype=bool)
    for i in range(len(points)):
        alarm[i] = int(points[max(0, i - window + 1):i + 1].sum()) >= required
    return alarm


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--visual", type=Path, required=True)
    parser.add_argument("--visual-scores", type=Path, required=True)
    parser.add_argument("--dynamics-scores", type=Path, required=True)
    parser.add_argument("--fusion", type=Path, required=True)
    parser.add_argument("--max-normal-episode-fp", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    visual = np.load(args.visual, allow_pickle=False)
    flow = np.load(args.visual_scores, allow_pickle=False)
    dynamics = np.load(args.dynamics_scores, allow_pickle=False)
    config = json.loads(args.fusion.read_text(encoding="utf-8"))
    if not (len(visual["visual"]) == len(flow["directed_response"]) == len(dynamics["normal_logp"])):
        raise ValueError("visual and dynamics rows do not align")

    state = classify_states(dynamics, config, visual["task_id"])
    valid = visual["valid"].astype(bool)
    active = visual["abnormal"].astype(bool)
    clean = np.isclose(visual["scale"].astype(float), 1.0)
    visual_evidence = -flow["directed_response"]
    novelty = np.zeros(len(state), dtype=float)
    for i, task_id in enumerate(visual["task_id"]):
        current = {**config, **config.get("by_task", {}).get(str(int(task_id)), {})}
        novelty[i] = (current["normal_logp_center"] - dynamics["normal_logp"][i]) / current["normal_logp_scale"]

    episodes: dict[str, list[int]] = collections.defaultdict(list)
    for i, episode_id in enumerate(visual["episode_id"]):
        episodes[str(episode_id)].append(i)
    for episode_id in episodes:
        episodes[episode_id].sort(key=lambda i: int(visual["action_index"][i]))

    candidate_values = visual_evidence[valid & (state == "ambiguous_overlap")]
    candidates = np.r_[np.unique(candidate_values), np.inf]

    def evaluate(threshold: float) -> dict:
        details = []
        active_alarms = inactive_alarms = active_count = inactive_count = 0
        for episode_id, indices in episodes.items():
            indices = np.asarray(indices)
            visual_point = (
                valid[indices]
                & (state[indices] == "ambiguous_overlap")
                & (visual_evidence[indices] >= threshold)
            )
            base_point = (state[indices] == "known_abnormal") | (
                (state[indices] == "unknown_abnormal")
                & (novelty[indices] >= config["unknown_novelty_threshold"])
            )
            base_alarm = persistent(base_point)
            combined_alarm = persistent(base_point | visual_point)
            is_active = active[indices]
            active_count += int(is_active.sum())
            inactive_count += int((~is_active).sum())
            active_alarms += int((combined_alarm & is_active).sum())
            inactive_alarms += int((combined_alarm & ~is_active).sum())
            details.append({
                "episode_id": episode_id,
                "clean": bool(clean[indices].all()),
                "base_detected": bool(np.any(base_alarm & is_active)),
                "combined_detected": bool(np.any(combined_alarm & is_active)),
                "base_any_alarm": bool(base_alarm.any()),
                "combined_any_alarm": bool(combined_alarm.any()),
            })
        return {
            "threshold": float(threshold),
            "normal_false_alarms": sum(d["combined_any_alarm"] for d in details if d["clean"]),
            "base_normal_false_alarms": sum(d["base_any_alarm"] for d in details if d["clean"]),
            "abnormal_detections": sum(d["combined_detected"] for d in details if not d["clean"]),
            "base_abnormal_detections": sum(d["base_detected"] for d in details if not d["clean"]),
            "active_step_recall": active_alarms / max(active_count, 1),
            "inactive_step_alarm_rate": inactive_alarms / max(inactive_count, 1),
            "details": details,
        }

    choices = []
    for threshold in candidates:
        report = evaluate(float(threshold))
        if report["normal_false_alarms"] <= args.max_normal_episode_fp:
            choices.append(report)
    best = max(
        choices,
        key=lambda r: (r["abnormal_detections"], r["active_step_recall"], -r["inactive_step_alarm_rate"]),
    )
    best.update({
        "calibration_only": True,
        "normal_episode_budget": args.max_normal_episode_fp,
        "visual_evidence": "negative directed optical-flow response",
        "visual_scope": "ambiguous_overlap only",
        "persistence": "2 of 3 points",
    })
    args.out.write_text(json.dumps(best, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in best.items() if k != "details"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
