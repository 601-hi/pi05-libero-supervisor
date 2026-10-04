#!/usr/bin/env python3
"""Small synthetic causality/shape smoke test; it is not a performance test."""
from __future__ import annotations

import numpy as np

from action_conditioned_visual_residual import fit, score


def main() -> None:
    rng = np.random.default_rng(4)
    n, h, w, c = 180, 4, 4, 32
    condition = rng.normal(size=(n, 68)).astype(np.float32)
    previous_base = rng.normal(size=(n, h, w, c)).astype(np.float32)
    previous_wrist = rng.normal(size=(n, h, w, c)).astype(np.float32)
    effect = condition[:, 0, None, None, None] * np.ones((1, h, w, c), np.float32) * 0.02
    base_delta = effect + rng.normal(scale=.002, size=(n, h, w, c)).astype(np.float32)
    wrist_delta = effect + rng.normal(scale=.002, size=(n, h, w, c)).astype(np.float32)
    data = {
        "condition_mean": condition, "base_current": previous_base + base_delta,
        "base_delta": base_delta, "wrist_current": previous_wrist + wrist_delta,
        "wrist_delta": wrist_delta, "scale": np.ones(n),
    }
    model = fit(data, projected_channels=4, ridge=1.0, seed=2)
    clean = score(model, data)["residual_score"]
    abnormal = dict(data)
    abnormal["base_delta"] = data["base_delta"].copy()
    abnormal["base_delta"][:, 0, 0] += 0.2
    abnormal["base_current"] = previous_base + abnormal["base_delta"]
    shifted = score(model, abnormal)["residual_score"]
    assert np.median(shifted) > np.median(clean) + 5.0
    print({"ok": True, "clean_median": float(np.median(clean)), "shifted_median": float(np.median(shifted))})


if __name__ == "__main__":
    main()
