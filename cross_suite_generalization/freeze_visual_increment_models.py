"""Freeze transparent action/visual/fusion baselines on development episodes only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


def auc(scores, labels):
    positive = scores[labels == 1]
    negative = scores[labels == 0]
    return float(sum((a > b) + .5 * (a == b) for a in positive for b in negative) / (len(positive) * len(negative)))


def fit_model(x, y, feature_names):
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale = np.maximum(scale, 1e-6)
    z = (x - mean) / scale
    counts = np.bincount(y, minlength=2)
    weights = np.asarray([len(y) / (2 * counts[label]) for label in y])
    def objective(parameters):
        logits = parameters[0] + z @ parameters[1:]
        loss = np.logaddexp(0, logits) - y * logits
        return float(np.sum(weights * loss) / np.sum(weights) + .5 * np.sum(parameters[1:] ** 2) / len(y))
    result = minimize(objective, np.zeros(z.shape[1] + 1), method="BFGS")
    if not result.success:
        raise RuntimeError(result.message)
    scores = expit(result.x[0] + z @ result.x[1:])
    threshold = float(np.percentile(scores[y == 0], 95))
    return {
        "feature_names": feature_names, "mean": mean.tolist(), "scale": scale.tolist(),
        "intercept": float(result.x[0]), "coefficients": result.x[1:].tolist(),
        "threshold_success_q95": threshold,
        "development_auc": auc(scores, y),
        "development_success_alarm_rate": float(np.mean(scores[y == 0] >= threshold)),
        "development_failure_detection_rate": float(np.mean(scores[y == 1] >= threshold)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--tracks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    tracks = json.loads(args.tracks.read_text(encoding="utf-8"))
    meta = {row["episode_id"]: row for row in manifest["episodes"]}
    measured = {(row["episode_id"], row["role"]): row for row in tracks["records"] if row["status"] == "measured"}
    rows = []
    for episode_id in sorted({key[0] for key in measured}):
        if (episode_id, "low_response") not in measured or (episode_id, "high_response") not in measured:
            continue
        episode = meta[episode_id]
        low = measured[(episode_id, "low_response")]["window_median_set_max_residual_px"]
        high = measured[(episode_id, "high_response")]["window_median_set_max_residual_px"]
        response = episode["label_blind_low_response_window"]["response_ratio"]
        rows.append({
            "episode_id": episode_id, "failure": not episode["success"],
            "log_response": float(np.log(response + 1e-4)),
            "log_low_visual": float(np.log(low + 1e-4)),
            "visual_log_drop": float(np.log(high + 1e-4) - np.log(low + 1e-4)),
        })
    y = np.asarray([row["failure"] for row in rows], dtype=int)
    specifications = {
        "action_only": ["log_response"],
        "visual_only": ["log_low_visual"],
        "action_visual_fusion": ["log_response", "log_low_visual", "visual_log_drop"],
    }
    models = {}
    for name, features in specifications.items():
        x = np.asarray([[row[field] for field in features] for row in rows], dtype=float)
        models[name] = fit_model(x, y, features)
    payload = {
        "schema_version": 1,
        "training_firewall": "development manifest and development visual tracks only",
        "episodes": len(rows), "failures": int(y.sum()), "successes": int((1-y).sum()),
        "regularization": "fixed L2 coefficient 1/N; no hyperparameter search",
        "models": models,
    }
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
