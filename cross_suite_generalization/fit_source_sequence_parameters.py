#!/usr/bin/env python3
"""Select sequential detector parameters on the reserved source-calibration half."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

from reliability_sequence_supervisor import ReliabilityCalibration, SequentialRiskAccumulator


def evaluate(data, calibration, params):
    episode_results = []
    unknown_steps = total_steps = 0
    for episode in np.unique(data["episode"]):
        mask = data["episode"] == episode
        order = np.argsort(data["action"][mask])
        indices = np.flatnonzero(mask)[order]
        label = bool(data["abnormal"][indices][0])
        accumulator = SequentialRiskAccumulator(**params)
        first_alarm = None
        for local_index, index in enumerate(indices):
            result = calibration.score(float(data["normal_logp"][index]), float(data["abnormal_logp"][index]), float(data["ensemble_std"][index]), 1.0)
            unknown_steps += result["decision"] == "unknown"; total_steps += 1
            sequence = accumulator.update(result["lr"], result["reliability"], result["decision"])
            if sequence["alarm"] and first_alarm is None: first_alarm = local_index
        episode_results.append((label, first_alarm))
    normal = [hit for label, hit in episode_results if not label]
    abnormal = [hit for label, hit in episode_results if label]
    return {
        "normal_episode_false_alarm": float(np.mean([x is not None for x in normal])),
        "abnormal_episode_detection": float(np.mean([x is not None for x in abnormal])),
        "median_detection_step": float(np.median([x for x in abnormal if x is not None])) if any(x is not None for x in abnormal) else None,
        "unknown_step_rate": unknown_steps / total_steps,
        "normal_episodes": len(normal), "abnormal_episodes": len(abnormal),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-normal-episode-false-alarm", type=float, default=.10)
    args = parser.parse_args()
    calibration = ReliabilityCalibration.from_json(args.calibration)
    data = np.load(args.scores)
    candidates = []
    for decay in (.80, .90, .95, .98):
        for drift in (0., .10, .25, .50):
            for alarm_threshold in (2., 3., 4., 6., 8.):
                params = {"alarm_threshold": alarm_threshold, "reset_threshold": alarm_threshold * .25,
                          "drift": drift, "decay": decay, "evidence_clip": 3., "unknown_patience": 3}
                metrics = evaluate(data, calibration, params)
                candidates.append({"parameters": params, "metrics": metrics})
    eligible = [x for x in candidates if x["metrics"]["normal_episode_false_alarm"] <= args.max_normal_episode_false_alarm]
    pool = eligible or candidates
    selected = max(pool, key=lambda x: (x["metrics"]["abnormal_episode_detection"],
                                        -(x["metrics"]["median_detection_step"] or 10**9),
                                        -x["metrics"]["normal_episode_false_alarm"]))
    payload = {"schema_version": 1, "source_only": True, "selection_constraint_met": bool(eligible),
               "constraint": {"max_normal_episode_false_alarm": args.max_normal_episode_false_alarm},
               "selected": selected, "candidate_count": len(candidates),
               "warning": None if eligible else "No candidate met the source normal-episode false-alarm constraint."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))

if __name__ == "__main__": main()
