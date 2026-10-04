#!/usr/bin/env python3
"""Evaluate a frozen, physics-gated supervisor on independent traces."""

import argparse
import json
import pathlib
import sys

import numpy as np

import evaluate_supervisor as evaluation
import normal_mechanism_analysis as normal_analysis


POINT_PERCENTILE = 97.5
MINIMUM_TARGET_M = 0.02
MAXIMUM_PROGRESS = 0.15
PERSISTENCE_REQUIRED = 2
PERSISTENCE_WINDOW = 3


def episode_outcomes(rows):
    return {
        (row["task_id"], row["episode_idx"]): bool(row["success"])
        for row in rows
        if row.get("event") == "episode_end"
    }


def fit_reference(path):
    rows = normal_analysis.load_jsonl(path)
    records, _, features, actual, groups = normal_analysis.build_dataset(rows)
    train, calibration, _ = normal_analysis.split_masks(groups)
    model = normal_analysis.fit_ridge(
        features[train], actual[train], features[calibration], actual[calibration]
    )
    prediction = normal_analysis.predict(model, features[calibration])
    covariance, _ = evaluation.covariance_inverse(actual[calibration] - prediction)
    score = evaluation.directional_shortfall(
        prediction, actual[calibration], prediction, covariance
    )
    threshold = float(np.percentile(score, POINT_PERCENTILE))
    return model, covariance, threshold


def evaluate_trace(path, model, covariance, threshold, anomaly):
    original = normal_analysis.load_jsonl(path)
    outcomes = episode_outcomes(original)
    records, _, features, actual, _ = normal_analysis.build_dataset(
        evaluation.force_include_all_episodes(original)
    )
    prediction = normal_analysis.predict(model, features)
    shortfall = evaluation.directional_shortfall(prediction, actual, prediction, covariance)
    point = (
        (shortfall > threshold)
        & (np.asarray([record["target_norm"] for record in records]) >= MINIMUM_TARGET_M)
        & (np.asarray([record["progress"] for record in records]) < MAXIMUM_PROGRESS)
    )
    alarm = evaluation.persistence_series(
        point.astype(float),
        records,
        0.5,
        required=PERSISTENCE_REQUIRED,
        window=PERSISTENCE_WINDOW,
    )
    groups = evaluation.episode_groups(records)
    alarm_events = []
    alarm_episodes = 0
    for key, indices in groups.items():
        hits = [index for index in indices if alarm[index]]
        if hits:
            alarm_episodes += 1
            alarm_events.append(
                {
                    "task_id": key[0],
                    "episode_idx": key[1],
                    "success": outcomes.get(key),
                    "first_action": records[hits[0]]["action_index"],
                    "alarm_steps": len(hits),
                }
            )
    result = {
        "path": path,
        "episodes": len(groups),
        "successful_episodes": int(sum(outcomes.values())),
        "steps": len(records),
        "alarm_steps": int(alarm.sum()),
        "step_alarm_rate": float(alarm.mean()),
        "alarm_episodes": alarm_episodes,
        "episode_alarm_rate": float(alarm_episodes / max(len(groups), 1)),
        "alarm_events": alarm_events,
    }
    if anomaly:
        active_lookup = {
            (row["task_id"], row["episode_idx"], row["action_index"]): bool(row.get("disturbance_active"))
            for row in original
            if row.get("event") == "step"
        }
        for record in records:
            record["disturbance_active"] = active_lookup[
                (record["task_id"], record["episode_idx"], record["action_index"])
            ]
        active = np.asarray([record["disturbance_active"] for record in records])
        result["active_steps"] = int(active.sum())
        result["active_step_recall"] = float(alarm[active].mean())
        result["detection"] = evaluation.detection_delays(records, alarm.astype(float), 0.5)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-normal", required=True)
    parser.add_argument("--evaluation-normal")
    parser.add_argument("--evaluation-anomaly", action="append", default=[])
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()
    model, covariance, threshold = fit_reference(args.reference_normal)
    report = {
        "frozen_rule": {
            "point_percentile": POINT_PERCENTILE,
            "point_threshold": threshold,
            "minimum_target_m": MINIMUM_TARGET_M,
            "maximum_progress": MAXIMUM_PROGRESS,
            "persistence_required": PERSISTENCE_REQUIRED,
            "persistence_window": PERSISTENCE_WINDOW,
        },
        "normal": (
            evaluate_trace(args.evaluation_normal, model, covariance, threshold, False)
            if args.evaluation_normal
            else None
        ),
        "anomalies": [
            evaluate_trace(path, model, covariance, threshold, True)
            for path in args.evaluation_anomaly
        ],
    }
    output = pathlib.Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
