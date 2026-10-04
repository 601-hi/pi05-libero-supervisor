"""Task-agnostic command-to-patch-optical-flow normal response model."""
from __future__ import annotations

import numpy as np


def fit(data, ridge: float = 10.0):
    valid = data["valid"].astype(bool)
    clean = np.isclose(data["scale"].astype(float), 1.0)
    selected = valid & clean
    x = data["x_condition"].astype(np.float64)[:, :-10]
    # 16 horizontal + 16 vertical Farneback-flow patch means.  Deliberately
    # exclude magnitude/difference and all appearance/context features.
    y = data["visual"].astype(np.float64)[:, :32]
    xm, xs = x[selected].mean(0), x[selected].std(0)
    xs[xs < 1e-8] = 1.0
    ym = y[selected].mean(0)
    xn = (x[selected] - xm) / xs
    yc = y[selected] - ym
    weights = np.linalg.solve(xn.T @ xn + ridge * np.eye(xn.shape[1]), xn.T @ yc)
    prediction = ym + xn @ weights
    residual = y[selected] - prediction
    residual_norm = np.linalg.norm(residual.reshape(len(residual), 2, 16), axis=1)
    center = np.median(residual_norm, axis=0)
    scale = 1.4826 * np.median(np.abs(residual_norm - center), axis=0)
    fallback = np.maximum(np.std(residual_norm, axis=0), 1e-6)
    scale[scale < 1e-6] = fallback[scale < 1e-6]
    return {"x_mean": xm, "x_scale": xs, "y_mean": ym, "weights": weights,
            "patch_center": center, "patch_scale": scale, "ridge": np.asarray(ridge)}


def score(model, data, top_k: int = 4):
    x = data["x_condition"].astype(np.float64)[:, :-10]
    actual = data["visual"].astype(np.float64)[:, :32]
    predicted = model["y_mean"] + ((x - model["x_mean"]) / model["x_scale"]) @ model["weights"]
    residual = actual - predicted
    patch_residual = np.linalg.norm(residual.reshape(len(residual), 2, 16), axis=1)
    patch_z = (patch_residual - model["patch_center"]) / model["patch_scale"]
    k = min(top_k, patch_z.shape[1])
    residual_score = np.partition(patch_z, -k, axis=1)[:, -k:].mean(1)
    an = np.linalg.norm(actual, axis=1)
    pn = np.linalg.norm(predicted, axis=1)
    cosine = np.sum(actual * predicted, axis=1) / (an * pn + 1e-12)
    response = np.sum(actual * predicted, axis=1) / (pn * pn + 1e-12)
    return {"residual_score": residual_score, "cosine": cosine,
            "directed_response": response, "actual_flow_norm": an, "predicted_flow_norm": pn}


def save(path, model):
    np.savez_compressed(path, **{k: np.asarray(v) for k, v in model.items()})
