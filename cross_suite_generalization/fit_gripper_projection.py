#!/usr/bin/env python3
"""Fit and cross-validate EEF-position to image-pixel projection models."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def affine_features(x: np.ndarray) -> np.ndarray:
    return np.column_stack((np.ones(len(x)), x))


def quadratic_features(x: np.ndarray) -> np.ndarray:
    a, b, c = x.T
    return np.column_stack((np.ones(len(x)), a, b, c, a*a, b*b, c*c, a*b, a*c, b*c))


def fit_ridge(features: np.ndarray, y: np.ndarray, ridge: float) -> np.ndarray:
    penalty = np.eye(features.shape[1]) * ridge
    penalty[0, 0] = 0.0
    return np.linalg.solve(features.T @ features + penalty, features.T @ y)


def evaluate_model(x: np.ndarray, y: np.ndarray, feature_fn, ridge: float) -> tuple[np.ndarray, np.ndarray]:
    predictions = []
    for held_out in range(len(x)):
        train = np.arange(len(x)) != held_out
        transform = fit_ridge(feature_fn(x[train]), y[train], ridge)
        predictions.append(feature_fn(x[held_out : held_out + 1]) @ transform)
    predictions = np.concatenate(predictions, axis=0)
    return predictions, np.linalg.norm(predictions - y, axis=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--pixels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--camera-id")
    parser.add_argument("--image-shape", type=int, nargs=2, metavar=("HEIGHT", "WIDTH"))
    parser.add_argument("--preprocessing-fingerprint")
    args = parser.parse_args()
    domain_values = (args.camera_id, args.image_shape, args.preprocessing_fingerprint)
    if any(value is not None for value in domain_values) and not all(value is not None for value in domain_values):
        parser.error("camera-id, image-shape and preprocessing-fingerprint must be supplied together")
    pairs = json.loads(args.pairs.read_text(encoding="utf-8"))["records"]
    pixels = {row["anonymous_id"]: row for row in json.loads(args.pixels.read_text(encoding="utf-8"))["records"]}
    ids = [row["anonymous_id"] for row in pairs]
    x = np.asarray([row["eef_position_xyz"] for row in pairs], dtype=float)
    y = np.asarray([pixels[identifier]["gripper_center_xy"] for identifier in ids], dtype=float)
    center, scale = x.mean(axis=0), x.std(axis=0)
    normalized = (x - center) / scale

    candidates = []
    for name, feature_fn in (("affine", affine_features), ("quadratic", quadratic_features)):
        for ridge in (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0):
            predicted, errors = evaluate_model(normalized, y, feature_fn, ridge)
            candidates.append({
                "model": name,
                "ridge": ridge,
                "loo_median_error_px": float(np.median(errors)),
                "loo_p90_error_px": float(np.percentile(errors, 90)),
                "loo_max_error_px": float(np.max(errors)),
                "per_episode": [
                    {"anonymous_id": identifier, "predicted_xy": predicted[index].tolist(), "error_px": float(errors[index])}
                    for index, identifier in enumerate(ids)
                ],
            })
    candidates.sort(key=lambda row: (row["loo_median_error_px"], row["loo_p90_error_px"]))
    best = candidates[0]
    feature_fn = affine_features if best["model"] == "affine" else quadratic_features
    transform = fit_ridge(feature_fn(normalized), y, best["ridge"])
    result = {
        "schema_version": 1,
        "selection_basis": "lowest leave-one-episode-out median pixel error; p90 is a small-sample diagnostic, not a hard safety bound",
        "calibration_warning": ("candidate selection and error reporting use the same samples; "
                                "independent grouped calibration is required for deployment"),
        "coordinate_normalization": {"mean": center.tolist(), "std": scale.tolist()},
        "best_model": {**{key: best[key] for key in ("model", "ridge", "loo_median_error_px", "loo_p90_error_px", "loo_max_error_px")}, "transform": transform.tolist()},
        "all_candidates": candidates,
    }
    if args.camera_id is not None:
        result["projection_domain"] = {
            "camera_id": args.camera_id,
            "image_shape": args.image_shape,
            "preprocessing_fingerprint": args.preprocessing_fingerprint,
        }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["best_model"], indent=2))
    worst = sorted(best["per_episode"], key=lambda row: row["error_px"], reverse=True)[:5]
    print("WORST_LOO", json.dumps(worst, indent=2))


if __name__ == "__main__":
    main()
