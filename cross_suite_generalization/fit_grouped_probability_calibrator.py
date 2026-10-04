"""Fit a small grouped logistic probability calibrator using development data only."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


ALLOWED_DEVELOPMENT_WAVES = {"Wave1", "Wave2"}


def sigmoid(value):
    value = np.clip(value, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-value))


def fit_logistic(x, y, *, l2=1e-2, steps=2000, learning_rate=.05):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-8] = 1.0
    z = (x - mean) / scale
    augmented = np.column_stack([np.ones(len(z)), z])
    weights = np.zeros(augmented.shape[1], dtype=np.float64)
    for _ in range(steps):
        probability = sigmoid(augmented @ weights)
        gradient = augmented.T @ (probability - y) / len(y)
        gradient[1:] += l2 * weights[1:]
        weights -= learning_rate * gradient
    return {"mean": mean, "scale": scale, "weights": weights}


def predict(model, x):
    x = np.asarray(x, dtype=np.float64)
    z = (x - model["mean"]) / model["scale"]
    return sigmoid(np.column_stack([np.ones(len(z)), z]) @ model["weights"])


def auc(y, probability):
    y = np.asarray(y, dtype=bool)
    positive = probability[y]
    negative = probability[~y]
    if not len(positive) or not len(negative):
        return None
    return float(np.mean(positive[:, None] > negative[None, :])
                 + .5 * np.mean(positive[:, None] == negative[None, :]))


def expected_calibration_error(y, probability, bins=10):
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(y)
    error = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (probability >= low) & (probability < high if high < 1 else probability <= high)
        if mask.any():
            error += mask.mean() * abs(float(probability[mask].mean()) - float(y[mask].mean()))
    return float(error) if total else None


def grouped_oof(x, y, groups, folds=5, **fit_kwargs):
    unique = sorted(set(str(group) for group in groups))
    if len(unique) < folds:
        raise ValueError("not enough episode groups for grouped cross-validation")
    assignment = {group: index % folds for index, group in enumerate(unique)}
    output = np.full(len(y), np.nan, dtype=np.float64)
    for fold in range(folds):
        test = np.asarray([assignment[str(group)] == fold for group in groups])
        train = ~test
        if len(set(y[train].tolist())) < 2:
            raise ValueError(f"fold {fold} training split lacks both classes")
        output[test] = predict(fit_logistic(x[train], y[train], **fit_kwargs), x[test])
    return output


def fit_artifact(x, y, groups, waves, *, feature_names, source_hash, folds=5):
    waves = {str(value) for value in waves}
    if not waves or not waves <= ALLOWED_DEVELOPMENT_WAVES:
        raise ValueError(f"only Wave1/Wave2 development data allowed, got {sorted(waves)}")
    y = np.asarray(y, dtype=np.int64)
    if set(y.tolist()) != {0, 1}:
        raise ValueError("labels must contain both binary classes")
    if min(int((y == 0).sum()), int((y == 1).sum())) < 5:
        raise ValueError("at least five examples from each class are required")
    oof = grouped_oof(np.asarray(x), y, groups, folds=folds)
    eps = 1e-8
    model = fit_logistic(np.asarray(x), y)
    return {
        "schema_version": 1,
        "model": "standardized_l2_logistic_probability_calibrator",
        "training_role": "Wave1/Wave2 development only",
        "source_sha256": source_hash,
        "feature_names": list(feature_names),
        "sample_count": int(len(y)),
        "episode_groups": int(len(set(str(group) for group in groups))),
        "class_counts": {"negative": int((y == 0).sum()), "positive": int((y == 1).sum())},
        "grouped_oof": {
            "folds": folds,
            "auc": auc(y, oof),
            "brier": float(np.mean((oof - y) ** 2)),
            "log_loss": float(-np.mean(y * np.log(oof + eps) + (1-y) * np.log(1-oof + eps))),
            "ece_10bin": expected_calibration_error(y, oof),
        },
        "parameters": {
            "mean": model["mean"].tolist(),
            "scale": model["scale"].tolist(),
            "intercept": float(model["weights"][0]),
            "coefficients": model["weights"][1:].tolist(),
        },
        "decision_policy": "probabilities are evidence; state-machine thresholds remain separately preregistered",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()
    raw = args.dataset.read_bytes()
    data = np.load(args.dataset, allow_pickle=False)
    required = {"features", "labels", "episode_id", "development_wave", "feature_names"}
    missing = required.difference(data.files)
    if missing:
        raise ValueError(f"dataset missing arrays: {sorted(missing)}")
    artifact = fit_artifact(
        data["features"], data["labels"], data["episode_id"], data["development_wave"],
        feature_names=[str(value) for value in data["feature_names"]],
        source_hash=hashlib.sha256(raw).hexdigest(), folds=args.folds,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(artifact["grouped_oof"], indent=2))


if __name__ == "__main__":
    main()
