#!/usr/bin/env python3
"""Analyze normal command-to-motion mismatch using deployable robot signals only."""

import argparse
import collections
import json
import pathlib

import numpy as np
from scipy import stats


DT = 0.05
EPS = 1e-12


def load_jsonl(path):
    with pathlib.Path(path).open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def quat_step_angle(q0, q1):
    q0 = np.asarray(q0, dtype=float)
    q1 = np.asarray(q1, dtype=float)
    denom = np.linalg.norm(q0) * np.linalg.norm(q1) + EPS
    cosine = np.clip(abs(float(np.dot(q0, q1))) / denom, 0.0, 1.0)
    return 2.0 * np.arccos(cosine)


def safe_cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + EPS))


def build_dataset(rows):
    success = {
        (row["task_id"], row["episode_idx"]): bool(row["success"])
        for row in rows
        if row.get("event") == "episode_end"
    }
    groups = collections.defaultdict(list)
    for row in rows:
        if row.get("event") != "step":
            continue
        key = (row["task_id"], row["episode_idx"])
        if success.get(key, False):
            groups[key].append(row)
    for values in groups.values():
        values.sort(key=lambda row: row["action_index"])

    records = []
    dynamic_features = []
    static_features = []
    outputs = []
    group_keys = []
    for key, steps in sorted(groups.items()):
        previous_action = np.zeros(7)
        previous_actual = np.zeros(3)
        for local_index, row in enumerate(steps):
            action = np.asarray(row["intended_action"], dtype=float)
            target = np.asarray(row["intended_target_translation"], dtype=float)
            actual = np.asarray(row["actual_translation"], dtype=float)
            qpos = np.asarray(row["joint_pos_before"], dtype=float)
            qvel = np.asarray(row["joint_vel_before"], dtype=float)
            eef_pos = np.asarray(row["eef_pos_before"], dtype=float)
            eef_quat = np.asarray(row["eef_quat_before"], dtype=float)
            grip_pos = np.asarray(row["gripper_qpos_before"], dtype=float)
            grip_vel = np.asarray(row["gripper_qvel_before"], dtype=float)
            action_delta = action - previous_action
            command_turn = safe_cosine(action[:3], previous_action[:3]) if local_index else 1.0
            velocity_alignment = safe_cosine(target, previous_actual) if local_index else 0.0
            target_norm = float(np.linalg.norm(target))
            actual_norm = float(np.linalg.norm(actual))
            progress = float(np.dot(target, actual) / (target_norm**2 + EPS))
            response = float(actual_norm / (target_norm + EPS))
            cosine = safe_cosine(target, actual)
            residual = float(np.linalg.norm(target - actual) / (target_norm + EPS))
            joint_speed = float(np.linalg.norm(qvel))
            previous_speed = float(np.linalg.norm(previous_actual) / DT)
            rotation_step = quat_step_angle(row["eef_quat_before"], row["eef_quat_after"])
            joint_velocity_delta = np.asarray(row["joint_vel_after"], dtype=float) - qvel
            eef_velocity_delta = (actual - previous_actual) / DT
            phase = int(row["action_index"] % 5)
            phase_onehot = np.eye(5, dtype=float)[phase]

            static = np.r_[target, target_norm]
            dynamic = np.r_[
                target,
                target_norm,
                action[3:6],
                action[6],
                qpos,
                qvel,
                eef_pos,
                eef_quat,
                grip_pos,
                grip_vel,
                previous_action,
                previous_actual,
                action_delta,
                command_turn,
                velocity_alignment,
                previous_speed,
                phase_onehot,
            ]
            records.append(
                {
                    "task_id": key[0],
                    "episode_idx": key[1],
                    "action_index": row["action_index"],
                    "target_norm": target_norm,
                    "actual_norm": actual_norm,
                    "progress": progress,
                    "response": response,
                    "cosine": cosine,
                    "normalized_residual": residual,
                    "joint_speed": joint_speed,
                    "previous_speed": previous_speed,
                    "command_turn": command_turn,
                    "velocity_alignment": velocity_alignment,
                    "eef_z": float(eef_pos[2]),
                    "rotation_action_norm": float(np.linalg.norm(action[3:6])),
                    "rotation_step": float(rotation_step),
                    "gripper_action": float(action[6]),
                    "gripper_speed": float(np.linalg.norm(grip_vel)),
                    "chunk_phase": phase,
                    "target_translation": target.tolist(),
                    "actual_translation": actual.tolist(),
                    "previous_actual_translation": previous_actual.tolist(),
                    "joint_velocity_delta": joint_velocity_delta.tolist(),
                    "eef_velocity_delta": eef_velocity_delta.tolist(),
                }
            )
            static_features.append(static)
            dynamic_features.append(dynamic)
            outputs.append(actual)
            group_keys.append(key)
            previous_action = action
            previous_actual = actual
    return (
        records,
        np.asarray(static_features),
        np.asarray(dynamic_features),
        np.asarray(outputs),
        group_keys,
    )


def percentiles(values):
    values = np.asarray(values, dtype=float)
    return {
        name: float(np.percentile(values, percentile))
        for name, percentile in (("p01", 1), ("p05", 5), ("p25", 25), ("p50", 50), ("p75", 75), ("p95", 95), ("p99", 99))
    }


def summarize_subset(records, indices):
    return {
        "count": int(len(indices)),
        "progress_median": float(np.median([records[i]["progress"] for i in indices])),
        "response_median": float(np.median([records[i]["response"] for i in indices])),
        "cosine_median": float(np.median([records[i]["cosine"] for i in indices])),
        "residual_median": float(np.median([records[i]["normalized_residual"] for i in indices])),
    }


def quantile_strata(records, feature):
    values = np.asarray([record[feature] for record in records])
    edges = np.unique(np.percentile(values, [0, 25, 50, 75, 100]))
    result = []
    for lower, upper in zip(edges[:-1], edges[1:]):
        indices = [
            i
            for i, value in enumerate(values)
            if value >= lower and (value < upper or (upper == edges[-1] and value <= upper))
        ]
        if indices:
            result.append({"range": [float(lower), float(upper)], **summarize_subset(records, indices)})
    return result


def categorical_strata(records, feature, categories):
    result = []
    for name, predicate in categories:
        indices = [i for i, record in enumerate(records) if predicate(record[feature])]
        if indices:
            result.append({"category": name, **summarize_subset(records, indices)})
    return result


def split_masks(group_keys):
    train = np.asarray([episode <= 2 for _, episode in group_keys])
    calibration = np.asarray([episode == 3 for _, episode in group_keys])
    test = np.asarray([episode >= 4 for _, episode in group_keys])
    return train, calibration, test


def fit_ridge(x_train, y_train, x_validation, y_validation):
    x_mean = x_train.mean(axis=0)
    x_scale = x_train.std(axis=0)
    x_scale[x_scale < 1e-8] = 1.0
    y_mean = y_train.mean(axis=0)
    y_scale = y_train.std(axis=0)
    y_scale[y_scale < 1e-8] = 1.0

    def transform_x(x):
        return np.c_[np.ones(len(x)), (x - x_mean) / x_scale]

    xtr = transform_x(x_train)
    xva = transform_x(x_validation)
    ytr = (y_train - y_mean) / y_scale
    best = None
    for ridge in (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0):
        penalty = np.eye(xtr.shape[1]) * ridge
        penalty[0, 0] = 0.0
        weights = np.linalg.solve(xtr.T @ xtr + penalty, xtr.T @ ytr)
        prediction = (xva @ weights) * y_scale + y_mean
        rmse = float(np.sqrt(np.mean((prediction - y_validation) ** 2)))
        if best is None or rmse < best[0]:
            best = (rmse, ridge, weights)
    return {
        "validation_rmse_m": best[0],
        "ridge": best[1],
        "weights": best[2],
        "x_mean": x_mean,
        "x_scale": x_scale,
        "y_mean": y_mean,
        "y_scale": y_scale,
    }


def predict(model, x):
    standardized = (x - model["x_mean"]) / model["x_scale"]
    design = np.c_[np.ones(len(x)), standardized]
    return (design @ model["weights"]) * model["y_scale"] + model["y_mean"]


def evaluate_prediction(actual, prediction):
    residual = actual - prediction
    rmse = float(np.sqrt(np.mean(residual**2)))
    baseline = float(np.sqrt(np.mean((actual - actual.mean(axis=0)) ** 2)))
    r2 = 1.0 - float(np.sum(residual**2) / (np.sum((actual - actual.mean(axis=0)) ** 2) + EPS))
    return {"rmse_m": rmse, "constant_rmse_m": baseline, "r2": r2}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-model", required=True)
    args = parser.parse_args()

    records, static_x, dynamic_x, y, group_keys = build_dataset(load_jsonl(args.trace))
    train, calibration, test = split_masks(group_keys)
    if not train.any() or not calibration.any() or not test.any():
        raise RuntimeError("Need successful episodes with indices 0-4 for train/calibration/test")

    report = {
        "trace": str(args.trace),
        "successful_episodes": len(set(group_keys)),
        "steps": len(records),
        "split_steps": {
            "train": int(train.sum()),
            "calibration": int(calibration.sum()),
            "test": int(test.sum()),
        },
        "outcome_percentiles": {
            name: percentiles([record[name] for record in records])
            for name in ("progress", "response", "cosine", "normalized_residual")
        },
        "spearman": {},
        "strata": {},
    }
    explanatory = (
        "target_norm",
        "joint_speed",
        "previous_speed",
        "command_turn",
        "velocity_alignment",
        "eef_z",
        "rotation_action_norm",
        "rotation_step",
        "gripper_action",
        "gripper_speed",
        "chunk_phase",
    )
    outcomes = ("progress", "response", "cosine", "normalized_residual")
    for feature in explanatory:
        report["spearman"][feature] = {}
        values = [record[feature] for record in records]
        for outcome in outcomes:
            coefficient, pvalue = stats.spearmanr(values, [record[outcome] for record in records])
            report["spearman"][feature][outcome] = {
                "rho": float(coefficient),
                "p": float(pvalue),
            }

    for feature in ("target_norm", "joint_speed", "previous_speed", "eef_z", "gripper_speed"):
        report["strata"][feature] = quantile_strata(records, feature)
    report["strata"]["command_turn"] = categorical_strata(
        records,
        "command_turn",
        (("reversal", lambda x: x < -0.5), ("turn", lambda x: -0.5 <= x < 0.5), ("aligned", lambda x: x >= 0.5)),
    )
    report["strata"]["velocity_alignment"] = categorical_strata(
        records,
        "velocity_alignment",
        (("opposing", lambda x: x < -0.3), ("cross_or_slow", lambda x: -0.3 <= x < 0.3), ("aligned", lambda x: x >= 0.3)),
    )
    report["strata"]["chunk_phase"] = categorical_strata(
        records, "chunk_phase", tuple((str(i), lambda x, i=i: x == i) for i in range(5))
    )
    report["strata"]["gripper_action"] = categorical_strata(
        records,
        "gripper_action",
        (("negative", lambda x: x < -0.5), ("neutral", lambda x: -0.5 <= x <= 0.5), ("positive", lambda x: x > 0.5)),
    )

    models = {}
    for name, x in (("static", static_x), ("dynamic", dynamic_x)):
        model = fit_ridge(x[train], y[train], x[calibration], y[calibration])
        prediction_calibration = predict(model, x[calibration])
        prediction_test = predict(model, x[test])
        report.setdefault("models", {})[name] = {
            "ridge": model["ridge"],
            "validation": evaluate_prediction(y[calibration], prediction_calibration),
            "test": evaluate_prediction(y[test], prediction_test),
        }
        models[name] = model

    dynamic_calibration_residual = y[calibration] - predict(models["dynamic"], dynamic_x[calibration])
    covariance = np.cov(dynamic_calibration_residual, rowvar=False) + np.eye(3) * 1e-10
    covariance_inverse = np.linalg.inv(covariance)
    dynamic_test_residual = y[test] - predict(models["dynamic"], dynamic_x[test])
    test_score = np.einsum("ni,ij,nj->n", dynamic_test_residual, covariance_inverse, dynamic_test_residual)
    report["dynamic_test_mahalanobis_percentiles"] = percentiles(test_score)

    output_json = pathlib.Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    dynamic = models["dynamic"]
    np.savez(
        args.output_model,
        weights=dynamic["weights"],
        x_mean=dynamic["x_mean"],
        x_scale=dynamic["x_scale"],
        y_mean=dynamic["y_mean"],
        y_scale=dynamic["y_scale"],
        covariance=covariance,
        covariance_inverse=covariance_inverse,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
