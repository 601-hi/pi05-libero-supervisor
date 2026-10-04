#!/usr/bin/env python3
"""Evaluate a fully frozen reliability/sequence supervisor on target scores."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

from reliability_sequence_supervisor import ReliabilityCalibration, SequentialRiskAccumulator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--sequence-parameters", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    calibration = ReliabilityCalibration.from_json(args.calibration)
    sequence_payload = json.loads(args.sequence_parameters.read_text(encoding="utf-8"))
    if not sequence_payload.get("source_only") or not sequence_payload.get("selection_constraint_met"):
        raise ValueError("sequence parameters are not an accepted source-only artifact")
    parameters = sequence_payload["selected"]["parameters"]
    data = np.load(args.scores)
    required = {"normal_logp", "abnormal_logp", "ensemble_std", "abnormal", "file_index", "episode_idx", "action_index"}
    if not required.issubset(data.files): raise ValueError(f"missing score fields: {sorted(required-set(data.files))}")
    episodes = []
    state_counts = {key: 0 for key in ("known_normal", "known_abnormal", "ambiguous_overlap", "unknown")}
    for file_index, episode_idx in sorted(set(zip(data["file_index"], data["episode_idx"]))):
        mask = (data["file_index"] == file_index) & (data["episode_idx"] == episode_idx)
        indices = np.flatnonzero(mask)[np.argsort(data["action_index"][mask])]
        accumulator = SequentialRiskAccumulator(**parameters)
        alarm_actions, protective_actions = [], []
        for index in indices:
            evidence = calibration.score(float(data["normal_logp"][index]), float(data["abnormal_logp"][index]), float(data["ensemble_std"][index]), 1.0)
            state_counts[evidence["decision"]] += 1
            result = accumulator.update(evidence["lr"], evidence["reliability"], evidence["decision"])
            if result["alarm"]: alarm_actions.append(int(data["action_index"][index]))
            if result["protective_observation"]: protective_actions.append(int(data["action_index"][index]))
        active = data["abnormal"][indices].astype(bool)
        onset = int(data["action_index"][indices][active].min()) if active.any() else None
        active_alarms = [x for x in alarm_actions if onset is not None and x >= onset and x <= int(data["action_index"][indices][active].max())]
        episodes.append({"file_index": int(file_index), "episode_idx": int(episode_idx), "abnormal": bool(active.any()),
                         "any_alarm": bool(alarm_actions), "active_alarm": bool(active_alarms),
                         "first_active_alarm_delay": active_alarms[0]-onset if active_alarms else None,
                         "protective_observation": bool(protective_actions)})
    normal = [x for x in episodes if not x["abnormal"]]; abnormal = [x for x in episodes if x["abnormal"]]
    total_states = sum(state_counts.values())
    result = {"frozen_evaluation": True, "episodes": len(episodes),
              "normal_episode_false_alarm": float(np.mean([x["any_alarm"] for x in normal])) if normal else None,
              "abnormal_episode_detection": float(np.mean([x["active_alarm"] for x in abnormal])) if abnormal else None,
              "protective_observation_episode_rate": float(np.mean([x["protective_observation"] for x in episodes])),
              "decision_coverage": 1.0-state_counts["unknown"]/max(total_states,1), "state_counts": state_counts,
              "episode_details": episodes}
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "episode_details"}, indent=2))

if __name__ == "__main__": main()
