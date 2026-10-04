#!/usr/bin/env python3
"""Audit a trained structured DROID response encoder on its sealed calibration split."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from train_external_response_encoder import INPUT_KEYS, TARGET_KEYS, normalized


def auc_rank(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(bool).reshape(-1)
    scores = scores.reshape(-1)
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        stop = start + 1
        while stop < len(scores) and sorted_scores[stop] == sorted_scores[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + 1 + stop)
        start = stop
    positives = int(labels.sum())
    negatives = len(labels) - positives
    return float((ranks[labels].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--readiness", type=Path, required=True)
    parser.add_argument("--data-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import torch
    from torch import nn

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    torch.set_num_threads(int(config.get("torch_num_threads", 1)))
    readiness = json.loads(args.readiness.read_text(encoding="utf-8"))
    manifest = json.loads((args.data_directory / "manifest.json").read_text(encoding="utf-8"))
    train = np.load(args.data_directory / manifest["output_files"]["train"]["path"], allow_pickle=False)
    calibration = np.load(
        args.data_directory / manifest["output_files"]["calibration"]["path"], allow_pickle=False
    )
    x_cal = normalized(calibration, INPUT_KEYS, readiness["normalization"])
    y_train = normalized(train, TARGET_KEYS, readiness["normalization"])
    y_cal = normalized(calibration, TARGET_KEYS, readiness["normalization"])
    output_dim = 2 * y_cal.shape[1] + 3
    model = nn.Sequential(
        nn.Linear(x_cal.shape[1], config["hidden_dim"]), nn.SiLU(),
        nn.Linear(config["hidden_dim"], config["hidden_dim"]), nn.SiLU(),
        nn.Linear(config["hidden_dim"], output_dim),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    with torch.no_grad():
        prediction = model(torch.from_numpy(x_cal)).numpy()

    dimension = y_cal.shape[1]
    mean = prediction[:, :dimension]
    log_variance = np.clip(prediction[:, dimension : 2 * dimension], -8.0, 6.0)
    standard_deviation = np.exp(0.5 * log_variance)
    residual = y_cal - mean
    z = residual / standard_deviation
    train_variance = np.maximum(np.mean(y_train ** 2, axis=0), np.exp(-8.0))
    model_nll = 0.5 * np.mean(log_variance + residual ** 2 / np.exp(log_variance))
    baseline_nll = 0.5 * np.mean(np.log(train_variance) + y_cal ** 2 / train_variance)

    raw_train_grip = train["response_gripper_delta"].reshape(-1)
    raw_cal_grip = calibration["response_gripper_delta"].reshape(-1)
    threshold = float(config.get("gripper_active_threshold", 1e-6))
    train_active = np.abs(raw_train_grip) > threshold
    cal_active = np.abs(raw_cal_grip) > threshold
    event_logit = prediction[:, 2 * dimension]
    event_probability = 1.0 / (1.0 + np.exp(-event_logit))
    event_decision = event_probability >= 0.5
    true_positive_rate = np.mean(event_decision[cal_active])
    true_negative_rate = np.mean(~event_decision[~cal_active])
    prior = float(train_active.mean())

    grip_center = float(checkpoint["summary"]["gripper_active_center"])
    grip_scale = float(checkpoint["summary"]["gripper_active_scale"])
    grip_target = (raw_cal_grip[cal_active] - grip_center) / grip_scale
    grip_mean = prediction[cal_active, 2 * dimension + 1]
    grip_log_variance = np.clip(prediction[cal_active, 2 * dimension + 2], -8.0, 6.0)
    grip_z = (grip_target - grip_mean) / np.exp(0.5 * grip_log_variance)

    result = {
        "checkpoint_summary": checkpoint["summary"],
        "calibration_samples": int(len(y_cal)),
        "continuous": {
            "model_nll": float(model_nll),
            "constant_baseline_nll": float(baseline_nll),
            "nll_improvement_fraction": float((baseline_nll - model_nll) / abs(baseline_nll)),
            "model_rmse": float(np.sqrt(np.mean(residual ** 2))),
            "constant_baseline_rmse": float(np.sqrt(np.mean(y_cal ** 2))),
            "rmse_improvement_fraction": float(
                (np.sqrt(np.mean(y_cal ** 2)) - np.sqrt(np.mean(residual ** 2)))
                / np.sqrt(np.mean(y_cal ** 2))
            ),
            "predicted_std_percentiles": dict(zip(
                ("p01", "p50", "p99"), map(float, np.percentile(standard_deviation, [1, 50, 99]))
            )),
            "standardized_residual_abs_coverage": {
                "le_1": float(np.mean(np.abs(z) <= 1)),
                "le_2": float(np.mean(np.abs(z) <= 2)),
                "le_3": float(np.mean(np.abs(z) <= 3)),
            },
        },
        "gripper_event": {
            "calibration_active_fraction": float(cal_active.mean()),
            "roc_auc": auc_rank(cal_active, event_probability),
            "balanced_accuracy_at_0_5": float(0.5 * (true_positive_rate + true_negative_rate)),
            "brier_score": float(np.mean((event_probability - cal_active.astype(float)) ** 2)),
            "constant_prior_brier_score": float(np.mean((prior - cal_active.astype(float)) ** 2)),
        },
        "gripper_active_magnitude": {
            "samples": int(cal_active.sum()),
            "model_rmse_normalized": float(np.sqrt(np.mean((grip_target - grip_mean) ** 2))),
            "constant_baseline_rmse_normalized": float(np.sqrt(np.mean(grip_target ** 2))),
            "standardized_residual_abs_coverage": {
                "le_1": float(np.mean(np.abs(grip_z) <= 1)),
                "le_2": float(np.mean(np.abs(grip_z) <= 2)),
                "le_3": float(np.mean(np.abs(grip_z) <= 3)),
            },
        },
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
