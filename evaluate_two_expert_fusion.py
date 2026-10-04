#!/usr/bin/env python3
"""Evaluate a frozen four-state fusion configuration on scored episodes."""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np

from two_expert_fusion import TwoExpertFusion


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(); d = np.load(args.scores, allow_pickle=False)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    groups = collections.defaultdict(list)
    for i, episode in enumerate(d["episode_id"]): groups[str(episode)].append(i)
    details = []
    state_counts = collections.Counter(); alarm_steps = np.zeros(len(d["normal_logp"]), dtype=bool)
    for episode, indices in groups.items():
        indices.sort(key=lambda i: int(d["action_index"][i])); fusion = TwoExpertFusion(config)
        first_active = next((j for j, i in enumerate(indices) if bool(d["abnormal"][i])), None)
        first_alarm = None; first_alarm_state = None
        active_state_counts = collections.Counter()
        for j, i in enumerate(indices):
            result = fusion.update(float(d["normal_logp"][i]), float(d["abnormal_logp"][i]), int(d["task_id"][i]))
            state_counts[result.state] += 1; alarm_steps[i] = result.persistent_alarm
            if bool(d["abnormal"][i]): active_state_counts[result.state] += 1
            if result.persistent_alarm and first_alarm is None:
                first_alarm = j; first_alarm_state = result.state
        has_active = first_active is not None
        detected_during_active = any(alarm_steps[i] and bool(d["abnormal"][i]) for i in indices)
        details.append({"episode_id": episode, "task_id": int(d["task_id"][indices[0]]),
                        "scale": float(d["scale"][indices[0]]),
                        "has_active_fault": has_active, "detected_during_active": detected_during_active,
                        "any_alarm": bool(alarm_steps[indices].any()),
                        "first_alarm_state": first_alarm_state,
                        "active_state_counts": dict(active_state_counts),
                        "delay_steps": None if first_active is None or first_alarm is None else first_alarm-first_active})
    active = d["abnormal"].astype(bool)
    normal_episodes = [e for e in details if not e["has_active_fault"]]
    abnormal_episodes = [e for e in details if e["has_active_fault"]]
    report = {"episodes": len(details), "state_step_counts": dict(state_counts),
              "normal_episode_false_alarms": sum(e["any_alarm"] for e in normal_episodes),
              "normal_episodes": len(normal_episodes),
              "abnormal_episode_detections": sum(e["detected_during_active"] for e in abnormal_episodes),
              "abnormal_episodes": len(abnormal_episodes),
              "active_step_recall": float(alarm_steps[active].mean()) if active.any() else None,
              "inactive_step_alarm_rate": float(alarm_steps[~active].mean()) if (~active).any() else None,
              "details": details}
    report["by_scale"] = {}
    for scale in sorted({e["scale"] for e in details}):
        subset=[e for e in details if e["scale"] == scale]
        report["by_scale"][str(scale)]={"episodes":len(subset),
            "detected_during_active":sum(e["detected_during_active"] for e in subset),
            "any_alarm":sum(e["any_alarm"] for e in subset)}
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k != "details"}, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
