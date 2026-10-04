#!/usr/bin/env python3
"""Evaluate deployable execution-deviation scores with episode-level splits."""

import argparse
import json
import pathlib

import numpy as np

import normal_mechanism_analysis as normal_analysis


EPS = 1e-12


def force_include_all_episodes(rows):
    copied = []
    for row in rows:
        row = dict(row)
        if row.get("event") == "episode_end":
            row["success"] = True
        copied.append(row)
    return copied


def mahalanobis(residual, inverse_covariance):
    return np.einsum("ni,ij,nj->n", residual, inverse_covariance, residual)


def covariance_inverse(residual):
    covariance = np.cov(residual, rowvar=False) + np.eye(residual.shape[1]) * 1e-10
    return covariance, np.linalg.inv(covariance)


def directional_shortfall(prediction, actual, direction, covariance):
    unit = direction / (np.linalg.norm(direction, axis=1, keepdims=True) + EPS)
    shortfall = np.sum((prediction - actual) * unit, axis=1)
    variance = np.einsum("ni,ij,nj->n", unit, covariance, unit)
    return shortfall / np.sqrt(variance + EPS)


def auroc(labels, scores):
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    positive = int(labels.sum())
    negative = int((~labels).sum())
    if positive == 0 or negative == 0:
        return None
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # Average tied ranks.
    unique, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    if np.any(counts > 1):
        for group in np.flatnonzero(counts > 1):
            tied = inverse == group
            ranks[tied] = ranks[tied].mean()
    rank_sum = ranks[labels].sum()
    return float((rank_sum - positive * (positive + 1) / 2) / (positive * negative))


def average_precision(labels, scores):
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    if labels.sum() == 0:
        return None
    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order]
    precision = np.cumsum(sorted_labels) / np.arange(1, len(labels) + 1)
    return float(precision[sorted_labels].mean())


def rolling_mean(scores, records, window):
    result = np.full(len(scores), np.nan)
    groups = {}
    for index, record in enumerate(records):
        groups.setdefault((record["task_id"], record["episode_idx"]), []).append(index)
    for indices in groups.values():
        indices.sort(key=lambda index: records[index]["action_index"])
        for offset, index in enumerate(indices):
            start = max(0, offset - window + 1)
            result[index] = float(np.mean(scores[indices[start : offset + 1]]))
    return result


def episode_groups(records):
    groups = {}
    for index, record in enumerate(records):
        groups.setdefault((record["task_id"], record["episode_idx"]), []).append(index)
    for indices in groups.values():
        indices.sort(key=lambda index: records[index]["action_index"])
    return groups


def cusum_series(scores, records, *, center, scale, drift):
    standardized = (np.asarray(scores) - center) / (scale + EPS)
    result = np.zeros(len(scores), dtype=float)
    for indices in episode_groups(records).values():
        accumulator = 0.0
        for index in indices:
            accumulator = max(0.0, accumulator + standardized[index] - drift)
            result[index] = accumulator
    return result


def episode_alarm_count(records, scores, threshold):
    return int(
        sum(any(scores[index] > threshold for index in indices) for indices in episode_groups(records).values())
    )


def persistence_series(scores, records, threshold, *, required, window):
    exceed = np.asarray(scores) > threshold
    result = np.zeros(len(scores), dtype=bool)
    for indices in episode_groups(records).values():
        for offset, index in enumerate(indices):
            start = max(0, offset - window + 1)
            result[index] = np.sum(exceed[indices[start : offset + 1]]) >= required
    return result


def detection_delays(records, scores, threshold):
    groups = {}
    for index, record in enumerate(records):
        groups.setdefault((record["task_id"], record["episode_idx"]), []).append(index)
    delays = []
    misses = 0
    pre_false_alarms = 0
    for indices in groups.values():
        indices.sort(key=lambda index: records[index]["action_index"])
        active = [index for index in indices if records[index]["disturbance_active"]]
        if not active:
            continue
        start_action = records[active[0]]["action_index"]
        end_action = records[active[-1]]["action_index"]
        pre_false_alarms += sum(
            scores[index] > threshold
            for index in indices
            if records[index]["action_index"] < start_action
        )
        detected = [
            records[index]["action_index"]
            for index in indices
            if start_action <= records[index]["action_index"] <= end_action and scores[index] > threshold
        ]
        if detected:
            delays.append(detected[0] - start_action)
        else:
            misses += 1
    return {
        "detected_episodes": len(delays),
        "missed_episodes": misses,
        "delay_steps_median": float(np.median(delays)) if delays else None,
        "delay_steps_p95": float(np.percentile(delays, 95)) if delays else None,
        "delay_ms_median": float(np.median(delays) * 50.0) if delays else None,
        "pre_disturbance_false_alarm_steps": int(pre_false_alarms),
    }


def context_matrix(records):
    """Mechanistic context available on a real robot before executing the current action."""
    return np.asarray(
        [
            [
                record["target_norm"],
                record["joint_speed"],
                record["previous_speed"],
                record["command_turn"],
                record["velocity_alignment"],
                record["eef_z"],
                record["rotation_action_norm"],
                record["gripper_speed"],
                record["gripper_action"],
                record["chunk_phase"],
            ]
            for record in records
        ],
        dtype=float,
    )


def local_quantile_ratio(
    reference_scores,
    reference_records,
    query_scores,
    query_records,
    *,
    neighbors,
    quantile,
    exclude_self=False,
):
    reference_context = context_matrix(reference_records)
    query_context = context_matrix(query_records)
    mean = reference_context.mean(axis=0)
    scale = reference_context.std(axis=0)
    scale[scale < 1e-8] = 1.0
    reference_context = (reference_context - mean) / scale
    query_context = (query_context - mean) / scale
    result = np.empty(len(query_scores), dtype=float)
    maximum_neighbors = len(reference_scores) - (1 if exclude_self else 0)
    k = min(neighbors, maximum_neighbors)
    for index, context in enumerate(query_context):
        distance = np.sum((reference_context - context) ** 2, axis=1)
        if exclude_self:
            distance[index] = np.inf
        nearest = np.argpartition(distance, k - 1)[:k]
        local_threshold = np.percentile(reference_scores[nearest], quantile)
        result[index] = query_scores[index] / (local_threshold + EPS)
    return result


def summarize_score(name, normal_calibration, normal_test, anomaly, labels, records):
    report = {"name": name, "windows": {}, "persistence": {}, "cusum": {}}
    for window in (1, 3, 5):
        calibration_window = rolling_mean(normal_calibration, records["normal_calibration"], window)
        normal_window = rolling_mean(normal_test, records["normal_test"], window)
        anomaly_window = rolling_mean(anomaly, records["anomaly"], window)
        threshold = float(np.percentile(calibration_window, 99))
        active = labels
        report["windows"][str(window)] = {
            "calibration_p99_threshold": threshold,
            "normal_test_fpr": float(np.mean(normal_window > threshold)),
            "active_step_recall": float(np.mean(anomaly_window[active] > threshold)),
            "auroc": auroc(active, anomaly_window),
            "average_precision": average_precision(active, anomaly_window),
            "detection": detection_delays(records["anomaly"], anomaly_window, threshold),
            "normal_test_score_percentiles": {
                "p50": float(np.percentile(normal_window, 50)),
                "p95": float(np.percentile(normal_window, 95)),
                "p99": float(np.percentile(normal_window, 99)),
            },
        }
    for percentile in (95.0, 97.5, 99.0):
        point_threshold = float(np.percentile(normal_calibration, percentile))
        for required, window in ((1, 1), (2, 3), (3, 5)):
            calibration_alarm = persistence_series(
                normal_calibration,
                records["normal_calibration"],
                point_threshold,
                required=required,
                window=window,
            )
            normal_alarm = persistence_series(
                normal_test,
                records["normal_test"],
                point_threshold,
                required=required,
                window=window,
            )
            anomaly_alarm = persistence_series(
                anomaly,
                records["anomaly"],
                point_threshold,
                required=required,
                window=window,
            )
            key = f"p{percentile:g}_{required}of{window}"
            report["persistence"][key] = {
                "point_threshold": point_threshold,
                "calibration_step_fpr": float(np.mean(calibration_alarm)),
                "normal_test_step_fpr": float(np.mean(normal_alarm)),
                "normal_test_alarm_episodes": episode_alarm_count(records["normal_test"], normal_alarm, 0.5),
                "normal_test_episodes": len(episode_groups(records["normal_test"])),
                "active_step_recall": float(np.mean(anomaly_alarm[labels])),
                "detection": detection_delays(records["anomaly"], anomaly_alarm.astype(float), 0.5),
            }
    center = float(np.median(normal_calibration))
    scale = float(np.median(np.abs(normal_calibration - center)) * 1.4826)
    if scale < 1e-8:
        scale = float(np.std(normal_calibration) + EPS)
    for drift in (0.0, 0.25, 0.5, 0.75, 1.0):
        calibration_cusum = cusum_series(
            normal_calibration, records["normal_calibration"], center=center, scale=scale, drift=drift
        )
        normal_cusum = cusum_series(normal_test, records["normal_test"], center=center, scale=scale, drift=drift)
        anomaly_cusum = cusum_series(anomaly, records["anomaly"], center=center, scale=scale, drift=drift)
        calibration_maxima = [
            max(calibration_cusum[index] for index in indices)
            for indices in episode_groups(records["normal_calibration"]).values()
        ]
        threshold = float(max(calibration_maxima))
        report["cusum"][str(drift)] = {
            "center": center,
            "robust_scale": scale,
            "threshold_max_calibration_episode": threshold,
            "normal_test_alarm_episodes": episode_alarm_count(records["normal_test"], normal_cusum, threshold),
            "normal_test_episodes": len(episode_groups(records["normal_test"])),
            "detection": detection_delays(records["anomaly"], anomaly_cusum, threshold),
        }
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--normal", required=True)
    parser.add_argument("--anomaly", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    normal_rows = normal_analysis.load_jsonl(args.normal)
    anomaly_original_rows = normal_analysis.load_jsonl(args.anomaly)
    normal_records, static_x, dynamic_x, normal_y, normal_groups = normal_analysis.build_dataset(normal_rows)
    anomaly_records, anomaly_static_x, anomaly_dynamic_x, anomaly_y, _ = normal_analysis.build_dataset(
        force_include_all_episodes(anomaly_original_rows)
    )
    train, calibration, test = normal_analysis.split_masks(normal_groups)

    active_lookup = {
        (row["task_id"], row["episode_idx"], row["action_index"]): bool(row.get("disturbance_active"))
        for row in anomaly_original_rows
        if row.get("event") == "step"
    }
    for record in anomaly_records:
        record["disturbance_active"] = active_lookup[
            (record["task_id"], record["episode_idx"], record["action_index"])
        ]

    anomaly_test = np.asarray([record["episode_idx"] >= 4 for record in anomaly_records])
    anomaly_records_test = [record for record, keep in zip(anomaly_records, anomaly_test) if keep]
    labels = np.asarray([record["disturbance_active"] for record in anomaly_records_test], dtype=bool)

    scores = {}
    for name, normal_x, anomaly_x in (
        ("static_ridge", static_x, anomaly_static_x),
        ("dynamic_ridge", dynamic_x, anomaly_dynamic_x),
    ):
        model = normal_analysis.fit_ridge(
            normal_x[train], normal_y[train], normal_x[calibration], normal_y[calibration]
        )
        calibration_residual = normal_y[calibration] - normal_analysis.predict(model, normal_x[calibration])
        covariance, inverse = covariance_inverse(calibration_residual)
        calibration_prediction = normal_analysis.predict(model, normal_x[calibration])
        test_prediction = normal_analysis.predict(model, normal_x[test])
        anomaly_prediction = normal_analysis.predict(model, anomaly_x[anomaly_test])
        scores[name] = {
            "calibration": mahalanobis(calibration_residual, inverse),
            "normal_test": mahalanobis(
                normal_y[test] - test_prediction, inverse
            ),
            "anomaly": mahalanobis(
                anomaly_y[anomaly_test] - anomaly_prediction, inverse
            ),
        }
        scores[f"{name}_target_shortfall"] = {
            "calibration": directional_shortfall(
                calibration_prediction, normal_y[calibration], normal_x[calibration, :3], covariance
            ),
            "normal_test": directional_shortfall(
                test_prediction, normal_y[test], normal_x[test, :3], covariance
            ),
            "anomaly": directional_shortfall(
                anomaly_prediction, anomaly_y[anomaly_test], anomaly_x[anomaly_test, :3], covariance
            ),
        }
        scores[f"{name}_predicted_shortfall"] = {
            "calibration": directional_shortfall(
                calibration_prediction, normal_y[calibration], calibration_prediction, covariance
            ),
            "normal_test": directional_shortfall(
                test_prediction, normal_y[test], test_prediction, covariance
            ),
            "anomaly": directional_shortfall(
                anomaly_prediction, anomaly_y[anomaly_test], anomaly_prediction, covariance
            ),
        }

    for name, record_field in (
        ("dynamic_eef_velocity_delta", "eef_velocity_delta"),
        ("dynamic_joint_velocity_delta", "joint_velocity_delta"),
    ):
        normal_output = np.asarray([record[record_field] for record in normal_records], dtype=float)
        anomaly_output = np.asarray([record[record_field] for record in anomaly_records], dtype=float)
        model = normal_analysis.fit_ridge(
            dynamic_x[train],
            normal_output[train],
            dynamic_x[calibration],
            normal_output[calibration],
        )
        calibration_prediction = normal_analysis.predict(model, dynamic_x[calibration])
        test_prediction = normal_analysis.predict(model, dynamic_x[test])
        anomaly_prediction = normal_analysis.predict(model, anomaly_dynamic_x[anomaly_test])
        calibration_residual = normal_output[calibration] - calibration_prediction
        _, inverse = covariance_inverse(calibration_residual)
        scores[name] = {
            "calibration": mahalanobis(calibration_residual, inverse),
            "normal_test": mahalanobis(normal_output[test] - test_prediction, inverse),
            "anomaly": mahalanobis(anomaly_output[anomaly_test] - anomaly_prediction, inverse),
        }

    normal_progress = np.asarray([record["progress"] for record in normal_records])
    anomaly_progress = np.asarray([record["progress"] for record in anomaly_records])
    scores["negative_progress"] = {
        "calibration": -normal_progress[calibration],
        "normal_test": -normal_progress[test],
        "anomaly": -anomaly_progress[anomaly_test],
    }

    record_sets = {
        "normal_calibration": [record for record, keep in zip(normal_records, calibration) if keep],
        "normal_test": [record for record, keep in zip(normal_records, test) if keep],
        "anomaly": anomaly_records_test,
    }

    dynamic_raw = scores["dynamic_ridge"]
    for neighbors, quantile in ((50, 95), (100, 95), (200, 95), (100, 99)):
        name = f"dynamic_local_k{neighbors}_q{quantile}"
        scores[name] = {
            "calibration": local_quantile_ratio(
                dynamic_raw["calibration"],
                record_sets["normal_calibration"],
                dynamic_raw["calibration"],
                record_sets["normal_calibration"],
                neighbors=neighbors,
                quantile=quantile,
                exclude_self=True,
            ),
            "normal_test": local_quantile_ratio(
                dynamic_raw["calibration"],
                record_sets["normal_calibration"],
                dynamic_raw["normal_test"],
                record_sets["normal_test"],
                neighbors=neighbors,
                quantile=quantile,
            ),
            "anomaly": local_quantile_ratio(
                dynamic_raw["calibration"],
                record_sets["normal_calibration"],
                dynamic_raw["anomaly"],
                record_sets["anomaly"],
                neighbors=neighbors,
                quantile=quantile,
            ),
        }
    report = {
        "normal_trace": args.normal,
        "anomaly_trace": args.anomaly,
        "normal_split_steps": {
            "train": int(train.sum()),
            "calibration": int(calibration.sum()),
            "test": int(test.sum()),
        },
        "anomaly_test_steps": int(anomaly_test.sum()),
        "anomaly_active_steps": int(labels.sum()),
        "scores": [],
    }
    for name, values in scores.items():
        report["scores"].append(
            summarize_score(
                name,
                values["calibration"],
                values["normal_test"],
                values["anomaly"],
                labels,
                record_sets,
            )
        )

    output = pathlib.Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
