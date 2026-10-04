"""Action-conditioned, patch-preserving visual residual baseline.

This model is intentionally a compact ridge predictor.  It predicts visual
latent change from the causally previous visual state and robot condition; it
never receives anomaly labels or task IDs.
"""
from __future__ import annotations

import numpy as np


def channel_projection(channel_count: int, projected_channels: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    matrix = rng.choice((-1.0, 1.0), size=(channel_count, projected_channels))
    return (matrix / np.sqrt(projected_channels)).astype(np.float32)


def project_patches(latent: np.ndarray, projection: np.ndarray) -> np.ndarray:
    if latent.ndim != 4:
        raise ValueError(f"latent must be [N,H,W,C], got {latent.shape}")
    if latent.shape[-1] != projection.shape[0]:
        raise ValueError("latent/projection channel mismatch")
    return np.einsum("nhwc,cp->nhwp", latent.astype(np.float32), projection, optimize=True)


def build_xy(data, projection: np.ndarray):
    condition = data["condition_mean"].astype(np.float32)
    # The final ten fields are the spatial-suite task one-hot.  Remove them so
    # the visual model cannot use task identity as a shortcut.
    if condition.shape[1] < 11:
        raise ValueError("condition vector is too short to contain task identity")
    condition = condition[:, :-10]
    camera_inputs = []
    camera_targets = []
    for name in ("base", "wrist"):
        current = data[f"{name}_current"].astype(np.float32)
        delta = data[f"{name}_delta"].astype(np.float32)
        previous = current - delta
        previous_projected = project_patches(previous, projection)
        delta_projected = project_patches(delta, projection)
        # Global first/second moments describe the previous visual state while
        # the output remains patch-localized.
        axes = (1, 2)
        summary = np.concatenate((previous_projected.mean(axes), previous_projected.std(axes)), axis=1)
        camera_inputs.append(summary)
        camera_targets.append(delta_projected.reshape(len(delta_projected), -1))
    x = np.concatenate((condition, *camera_inputs), axis=1)
    y = np.concatenate(camera_targets, axis=1)
    return x, y


def fit(data, projected_channels: int = 16, ridge: float = 10.0, seed: int = 20260905):
    projection = channel_projection(data["base_current"].shape[-1], projected_channels, seed)
    x, y = build_xy(data, projection)
    clean = np.isclose(data["scale"].astype(float), 1.0)
    if clean.sum() < x.shape[1] + 2:
        raise ValueError(f"need more clean rows than input dimensions ({clean.sum()} <= {x.shape[1] + 1})")
    xm, xs = x[clean].mean(0), x[clean].std(0)
    xs[xs < 1e-6] = 1.0
    ym = y[clean].mean(0)
    xn = (x[clean] - xm) / xs
    yc = y[clean] - ym
    weights = np.linalg.solve(xn.T @ xn + ridge * np.eye(xn.shape[1]), xn.T @ yc)
    residual = y[clean] - (ym + xn @ weights)
    patch_width = projected_channels
    patch_error = np.sqrt(np.mean(residual.reshape(len(residual), -1, patch_width) ** 2, axis=2))
    center = np.median(patch_error, axis=0)
    scale = 1.4826 * np.median(np.abs(patch_error - center), axis=0)
    scale[scale < 1e-6] = np.maximum(np.std(patch_error, axis=0)[scale < 1e-6], 1e-6)
    return {"projection": projection, "x_mean": xm, "x_scale": xs, "y_mean": ym,
            "weights": weights, "patch_center": center, "patch_scale": scale,
            "projected_channels": np.asarray(projected_channels), "ridge": np.asarray(ridge),
            "seed": np.asarray(seed)}


def score(model, data, top_k: int = 4):
    x, y = build_xy(data, model["projection"])
    prediction = model["y_mean"] + ((x - model["x_mean"]) / model["x_scale"]) @ model["weights"]
    width = int(model["projected_channels"])
    residual = y - prediction
    patch_error = np.sqrt(np.mean(residual.reshape(len(residual), -1, width) ** 2, axis=2))
    patch_z = (patch_error - model["patch_center"]) / model["patch_scale"]
    k = min(top_k, patch_z.shape[1])
    top = np.partition(patch_z, -k, axis=1)[:, -k:]
    residual_score = top.mean(axis=1)
    yn = np.linalg.norm(y, axis=1)
    pn = np.linalg.norm(prediction, axis=1)
    cosine = np.sum(y * prediction, axis=1) / (yn * pn + 1e-12)
    return {"residual_score": residual_score, "cosine": cosine,
            "actual_change_norm": yn, "predicted_change_norm": pn}


def save(path, model) -> None:
    np.savez_compressed(path, **{key: np.asarray(value) for key, value in model.items()})


def load(path):
    with np.load(path, allow_pickle=False) as values:
        return {key: values[key] for key in values.files}
