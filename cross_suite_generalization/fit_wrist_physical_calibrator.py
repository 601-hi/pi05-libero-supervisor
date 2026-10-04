#!/usr/bin/env python3
"""Fit an episode-grouped Platt calibrator for wrist physical scores."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def sigmoid(value: np.ndarray) -> np.ndarray:
    value = np.clip(value, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-value))


def fit_logistic(x: np.ndarray, y: np.ndarray, l2: float = 1.0, iterations: int = 100) -> np.ndarray:
    design = np.column_stack([np.ones(len(x)), x])
    weights = np.zeros(2, dtype=float)
    penalty = np.diag([0.0, l2])
    for _ in range(iterations):
        probability = sigmoid(design @ weights)
        gradient = design.T @ (probability - y) + penalty @ weights
        curvature = probability * (1.0 - probability)
        hessian = design.T @ (design * curvature[:, None]) + penalty + 1e-8 * np.eye(2)
        update = np.linalg.solve(hessian, gradient)
        weights -= update
        if np.linalg.norm(update) < 1e-10:
            break
    return weights


def auc(y: np.ndarray, probability: np.ndarray) -> float:
    positive = probability[y == 1]
    negative = probability[y == 0]
    return float(np.mean([(left > right) + 0.5 * (left == right) for left in positive for right in negative]))


def metrics(y: np.ndarray, probability: np.ndarray, bins: int = 10) -> dict:
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    occupied = []
    for index in range(bins):
        selected = (probability >= edges[index]) & (
            probability <= edges[index + 1] if index == bins - 1 else probability < edges[index + 1]
        )
        if selected.any():
            confidence = float(probability[selected].mean())
            frequency = float(y[selected].mean())
            ece += float(selected.mean()) * abs(confidence - frequency)
            occupied.append({"count": int(selected.sum()), "mean_probability": confidence, "positive_fraction": frequency})
    return {
        "count": int(len(y)),
        "positive_count": int(y.sum()),
        "auc": auc(y, probability),
        "brier": float(np.mean((probability - y) ** 2)),
        "log_loss": float(-np.mean(y * np.log(probability + 1e-12) + (1-y) * np.log(1-probability + 1e-12))),
        "ece_10_bin": float(ece),
        "occupied_bins": occupied,
    }


def dataset(fusion: dict, annotations: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scores, labels, groups = [], [], []
    for episode in fusion["episodes"]:
        identifier = episode["anonymous_id"]
        acceptable = set(annotations[identifier]["acceptable_candidate_ids"])
        for candidate in episode["candidates"]:
            scores.append(float(candidate["physical_fusion_score"]))
            labels.append(int(candidate["candidate_id"] in acceptable))
            groups.append(identifier)
    return np.asarray(scores), np.asarray(labels, dtype=int), np.asarray(groups)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--development-fusion", type=Path, required=True)
    parser.add_argument("--development-annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--external-fusion", type=Path)
    parser.add_argument("--external-annotations", type=Path)
    args = parser.parse_args()
    development = json.loads(args.development_fusion.read_text(encoding="utf-8"))
    annotations = json.loads(args.development_annotations.read_text(encoding="utf-8"))["annotations"]
    x, y, groups = dataset(development, annotations)
    mean, scale = float(x.mean()), float(x.std() or 1.0)
    normalized = (x - mean) / scale
    probabilities = np.zeros(len(x), dtype=float)
    for group in sorted(set(groups)):
        train = groups != group
        test = ~train
        weights = fit_logistic(normalized[train], y[train])
        probabilities[test] = sigmoid(np.column_stack([np.ones(test.sum()), normalized[test]]) @ weights)
    final_weights = fit_logistic(normalized, y)
    result = {
        "schema_version": 1,
        "development_only": True,
        "calibration_target": "candidate is an acceptable physically controlled-object mask",
        "grouping": "leave-one-episode-out",
        "feature": "physical_fusion_score",
        "normalization": {"mean": mean, "scale": scale},
        "model": {"intercept": float(final_weights[0]), "coefficient": float(final_weights[1]), "l2": 1.0},
        "cross_validated_development_metrics": metrics(y, probabilities),
    }
    if args.external_fusion or args.external_annotations:
        if not args.external_fusion or not args.external_annotations:
            raise ValueError("external-fusion and external-annotations must be supplied together")
        external = json.loads(args.external_fusion.read_text(encoding="utf-8"))
        external_annotations = json.loads(args.external_annotations.read_text(encoding="utf-8"))["annotations"]
        external_x, external_y, _ = dataset(external, external_annotations)
        external_probability = sigmoid(final_weights[0] + final_weights[1] * ((external_x - mean) / scale))
        result["retrospective_external_metrics"] = metrics(external_y, external_probability)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
