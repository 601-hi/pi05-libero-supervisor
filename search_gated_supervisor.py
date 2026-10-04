#!/usr/bin/env python3
"""Search interpretable persistence gates on validation/test episode splits."""

import sys

import numpy as np

import evaluate_supervisor as evaluation
import normal_mechanism_analysis as normal_analysis


NORMAL = "/root/gpufree-data/libero-traces/normal_kinematics_spatial_5ep_seed7_noise20260901.jsonl"
ANOMALY_TEMPLATE = (
    "/root/gpufree-data/libero-traces/"
    "disturb_scale{}_start40_len10_spatial_5ep_seed7_noise20260901.jsonl"
)


def anomaly_dataset(path, model, covariance):
    original = normal_analysis.load_jsonl(path)
    records, _, features, actual, _ = normal_analysis.build_dataset(
        evaluation.force_include_all_episodes(original)
    )
    keep = np.asarray([record["episode_idx"] >= 4 for record in records])
    records = [record for record, selected in zip(records, keep) if selected]
    prediction = normal_analysis.predict(model, features[keep])
    score = evaluation.directional_shortfall(prediction, actual[keep], prediction, covariance)
    lookup = {
        (row["task_id"], row["episode_idx"], row["action_index"]): bool(row.get("disturbance_active"))
        for row in original
        if row.get("event") == "step"
    }
    for record in records:
        record["disturbance_active"] = lookup[
            (record["task_id"], record["episode_idx"], record["action_index"])
        ]
    return records, score


def main():
    records, _, features, actual, groups = normal_analysis.build_dataset(normal_analysis.load_jsonl(NORMAL))
    train, calibration, test = normal_analysis.split_masks(groups)
    model = normal_analysis.fit_ridge(
        features[train], actual[train], features[calibration], actual[calibration]
    )
    calibration_prediction = normal_analysis.predict(model, features[calibration])
    test_prediction = normal_analysis.predict(model, features[test])
    covariance, _ = evaluation.covariance_inverse(actual[calibration] - calibration_prediction)
    calibration_score = evaluation.directional_shortfall(
        calibration_prediction, actual[calibration], calibration_prediction, covariance
    )
    test_score = evaluation.directional_shortfall(test_prediction, actual[test], test_prediction, covariance)
    test_records = [record for record, selected in zip(records, test) if selected]
    point_threshold = float(np.percentile(calibration_score, 97.5))
    anomalies = {
        scale: anomaly_dataset(ANOMALY_TEMPLATE.format(scale), model, covariance)
        for scale in ("025", "050", "075")
    }
    print("POINT_THRESHOLD", point_threshold)
    for minimum_target in (0.0, 0.01, 0.015, 0.02, 0.025, 0.03):
        for maximum_progress in (1e9, 0.20, 0.15, 0.10):
            normal_point = (
                (test_score > point_threshold)
                & (np.asarray([record["target_norm"] for record in test_records]) >= minimum_target)
                & (np.asarray([record["progress"] for record in test_records]) < maximum_progress)
            )
            normal_alarm = evaluation.persistence_series(
                normal_point.astype(float), test_records, 0.5, required=2, window=3
            )
            normal_episode_alarms = evaluation.episode_alarm_count(test_records, normal_alarm, 0.5)
            if normal_episode_alarms > 1:
                continue
            anomaly_results = {}
            for scale, (anomaly_records, anomaly_score) in anomalies.items():
                point = (
                    (anomaly_score > point_threshold)
                    & (np.asarray([record["target_norm"] for record in anomaly_records]) >= minimum_target)
                    & (np.asarray([record["progress"] for record in anomaly_records]) < maximum_progress)
                )
                alarm = evaluation.persistence_series(
                    point.astype(float), anomaly_records, 0.5, required=2, window=3
                )
                anomaly_results[scale] = evaluation.detection_delays(
                    anomaly_records, alarm.astype(float), 0.5
                )
            print(
                {
                    "minimum_target": minimum_target,
                    "maximum_progress": maximum_progress,
                    "normal_alarm_episodes": normal_episode_alarms,
                    "normal_step_fpr": float(normal_alarm.mean()),
                    "anomalies": anomaly_results,
                }
            )


if __name__ == "__main__":
    main()
