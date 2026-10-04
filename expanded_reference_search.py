#!/usr/bin/env python3
"""Develop a broader normal reference while preserving episode-level calibration/test splits."""

import numpy as np

import evaluate_supervisor as evaluation
import normal_mechanism_analysis as normal_analysis


BASE = "/root/gpufree-data/libero-traces/normal_kinematics_spatial_5ep_seed7_noise20260901.jsonl"
EXPANDED = "/root/gpufree-data/libero-traces/normal_independent_spatial_10ep_seed11_noise20260902.jsonl"


def load_all(path):
    return normal_analysis.build_dataset(
        evaluation.force_include_all_episodes(normal_analysis.load_jsonl(path))
    )


def main():
    base_records, _, base_x, base_y, _ = load_all(BASE)
    records, _, x, y, _ = load_all(EXPANDED)
    expanded_episode = np.asarray([record["episode_idx"] for record in records])
    development = expanded_episode <= 5
    calibration = (expanded_episode >= 6) & (expanded_episode <= 7)
    test = expanded_episode >= 8
    train_x = np.r_[base_x, x[development]]
    train_y = np.r_[base_y, y[development]]
    model = normal_analysis.fit_ridge(train_x, train_y, x[calibration], y[calibration])
    calibration_prediction = normal_analysis.predict(model, x[calibration])
    test_prediction = normal_analysis.predict(model, x[test])
    covariance, _ = evaluation.covariance_inverse(y[calibration] - calibration_prediction)
    calibration_score = evaluation.directional_shortfall(
        calibration_prediction, y[calibration], calibration_prediction, covariance
    )
    test_score = evaluation.directional_shortfall(test_prediction, y[test], test_prediction, covariance)
    calibration_records = [record for record, keep in zip(records, calibration) if keep]
    test_records = [record for record, keep in zip(records, test) if keep]
    print(
        "SPLIT_STEPS",
        {"base": len(base_records), "development": int(development.sum()), "calibration": int(calibration.sum()), "test": int(test.sum())},
    )
    print("MODEL_RIDGE", model["ridge"])
    print("TEST_PREDICTION", normal_analysis.evaluate_prediction(y[test], test_prediction))
    candidates = []
    for percentile in (95.0, 97.5, 99.0, 99.5, 99.75, 99.9):
        threshold = float(np.percentile(calibration_score, percentile))
        for minimum_target in (0.015, 0.02, 0.025, 0.03):
            for maximum_progress in (0.10, 0.125, 0.15, 0.175, 0.20):
                point = (
                    (test_score > threshold)
                    & (np.asarray([record["target_norm"] for record in test_records]) >= minimum_target)
                    & (np.asarray([record["progress"] for record in test_records]) < maximum_progress)
                )
                alarm = evaluation.persistence_series(
                    point.astype(float), test_records, 0.5, required=2, window=3
                )
                alarm_episodes = evaluation.episode_alarm_count(test_records, alarm, 0.5)
                candidates.append((alarm_episodes, float(alarm.mean()), percentile, minimum_target, maximum_progress, threshold, alarm))
                if alarm_episodes <= 1:
                    print(
                        {
                            "percentile": percentile,
                            "minimum_target": minimum_target,
                            "maximum_progress": maximum_progress,
                            "threshold": threshold,
                            "test_alarm_episodes": alarm_episodes,
                            "test_episodes": len(evaluation.episode_groups(test_records)),
                            "test_step_fpr": float(alarm.mean()),
                        }
                    )
    candidates.sort(key=lambda row: (row[0], row[1]))
    print("BEST_NORMAL_ONLY_CANDIDATES")
    for candidate in candidates[:10]:
        alarm_episodes, step_fpr, percentile, minimum_target, maximum_progress, threshold, _ = candidate
        print({
            "alarm_episodes": alarm_episodes,
            "test_episodes": len(evaluation.episode_groups(test_records)),
            "step_fpr": step_fpr,
            "percentile": percentile,
            "minimum_target": minimum_target,
            "maximum_progress": maximum_progress,
            "threshold": threshold,
        })
    best = candidates[0]
    best_alarm = best[-1]
    print("BEST_FALSE_ALARM_EPISODES")
    for key, indices in evaluation.episode_groups(test_records).items():
        active = [index for index in indices if best_alarm[index] > 0.5]
        if not active:
            continue
        print("EPISODE", key, "ALARM_STEPS", [test_records[index]["step_idx"] for index in active])
        for index in active[:8]:
            record = test_records[index]
            feature = x[test][index]
            print({
                "step": record["step_idx"],
                "chunk_position": record.get("chunk_position"),
                "score": float(test_score[index]),
                "target_norm": record["target_norm"],
                "progress": record["progress"],
                "response": record["response"],
                "cosine": record["cosine"],
                "previous_speed": record.get("previous_speed"),
                "velocity_alignment": record.get("velocity_alignment"),
                "action_delta_norm": record.get("action_delta_norm"),
                "gripper_action": record.get("action", [None] * 7)[6],
            })


if __name__ == "__main__":
    main()
