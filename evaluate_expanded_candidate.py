#!/usr/bin/env python3
"""Freeze a normal-only expanded reference and evaluate independent disturbances."""

import json
import pathlib

import numpy as np

import evaluate_supervisor as evaluation
import normal_mechanism_analysis as normal_analysis


BASE = "/root/gpufree-data/libero-traces/normal_kinematics_spatial_5ep_seed7_noise20260901.jsonl"
EXPANDED = "/root/gpufree-data/libero-traces/normal_independent_spatial_10ep_seed11_noise20260902.jsonl"
ANOMALIES = [
    "/root/gpufree-data/libero-traces/independent_disturb_scale025_start40_len10_spatial_3ep_seed11_noise20260902.jsonl",
    "/root/gpufree-data/libero-traces/independent_disturb_scale050_start40_len10_spatial_3ep_seed11_noise20260902.jsonl",
    "/root/gpufree-data/libero-traces/independent_disturb_scale075_start40_len10_spatial_3ep_seed11_noise20260902.jsonl",
]
OUT = "/root/gpufree-data/supervisor-results/expanded_candidate_independent_anomalies.json"


def dataset(path):
    rows = normal_analysis.load_jsonl(path)
    return rows, normal_analysis.build_dataset(evaluation.force_include_all_episodes(rows))


def evaluate(path, model, covariance, threshold):
    rows, (records, _, features, actual, _) = dataset(path)
    prediction = normal_analysis.predict(model, features)
    score = evaluation.directional_shortfall(prediction, actual, prediction, covariance)
    point = (
        (score > threshold)
        & (np.asarray([r["target_norm"] for r in records]) >= 0.02)
        & (np.asarray([r["progress"] for r in records]) < 0.15)
    )
    alarm = evaluation.persistence_series(point.astype(float), records, 0.5, required=2, window=3)
    active_lookup = {
        (r["task_id"], r["episode_idx"], r["action_index"]): bool(r.get("disturbance_active"))
        for r in rows if r.get("event") == "step"
    }
    active = np.asarray([
        active_lookup[(r["task_id"], r["episode_idx"], r["action_index"])] for r in records
    ])
    for r, flag in zip(records, active):
        r["disturbance_active"] = bool(flag)
    groups = evaluation.episode_groups(records)
    outside = ~active
    return {
        "path": path,
        "episodes": len(groups),
        "steps": len(records),
        "active_steps": int(active.sum()),
        "alarm_steps": int(alarm.sum()),
        "active_step_recall": float(alarm[active].mean()),
        "outside_step_alarm_rate": float(alarm[outside].mean()),
        "any_alarm_episodes": evaluation.episode_alarm_count(records, alarm, 0.5),
        "detection": evaluation.detection_delays(records, alarm.astype(float), 0.5),
    }


def main():
    _, (base_records, _, base_x, base_y, _) = dataset(BASE)
    _, (records, _, x, y, _) = dataset(EXPANDED)
    episode = np.asarray([r["episode_idx"] for r in records])
    development = episode <= 5
    calibration = (episode >= 6) & (episode <= 7)
    model = normal_analysis.fit_ridge(
        np.r_[base_x, x[development]], np.r_[base_y, y[development]],
        x[calibration], y[calibration],
    )
    calibration_prediction = normal_analysis.predict(model, x[calibration])
    covariance, _ = evaluation.covariance_inverse(y[calibration] - calibration_prediction)
    calibration_score = evaluation.directional_shortfall(
        calibration_prediction, y[calibration], calibration_prediction, covariance
    )
    threshold = float(np.percentile(calibration_score, 99.75))
    report = {
        "frozen": {"percentile": 99.75, "threshold": threshold, "minimum_target_m": 0.02,
                   "maximum_progress": 0.15, "persistence": "2_of_3"},
        "anomalies": [evaluate(path, model, covariance, threshold) for path in ANOMALIES],
    }
    output = pathlib.Path(OUT)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
