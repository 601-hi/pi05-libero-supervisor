#!/usr/bin/env python3
"""Train a leakage-controlled abnormal response expert from exported NPZ arrays.

The model is a conditional diagonal Gaussian mixture.  Labels such as scale and
disturbance onset are used only to select training rows and to report strata;
they are never part of x.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from dynamics_feature_schema import CONDITION_DIM_WITH_TASK, schema_sha256


class AbnormalMixtureExpert(torch.nn.Module):
    def __init__(self, input_dim: int, components: int = 3):
        super().__init__()
        self.components = components
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 128), torch.nn.SiLU(),
            torch.nn.Linear(128, 128), torch.nn.SiLU(),
            torch.nn.Linear(128, components * 7),
        )

    def forward(self, x: torch.Tensor):
        raw = self.net(x).reshape(-1, self.components, 7)
        return raw[..., 0], raw[..., 1:4], raw[..., 4:7].clamp(-7.0, 3.0)


def mixture_nll(logits, mean, logvar, target):
    residual = target[:, None, :] - mean
    log_component = -0.5 * (logvar + residual.square() * torch.exp(-logvar)).sum(-1)
    log_component -= 1.5 * np.log(2.0 * np.pi)
    return -torch.logsumexp(torch.log_softmax(logits, -1) + log_component, -1).mean()


def episode_split(episode_ids, validation_fraction: float, seed: int):
    unique = np.unique(episode_ids)
    rng = np.random.default_rng(seed); rng.shuffle(unique)
    n_validation = max(1, int(round(len(unique) * validation_fraction)))
    validation_episodes = set(unique[:n_validation].tolist())
    validation = np.asarray([value in validation_episodes for value in episode_ids])
    return ~validation, validation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--components", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--seed", type=int, default=20260903)
    args = parser.parse_args()
    data = np.load(args.dataset, allow_pickle=False)
    x_all, y_all = data["x"].astype(np.float64), data["y"].astype(np.float64)
    abnormal = data["abnormal"].astype(bool)
    if x_all.ndim != 2 or x_all.shape[1] != CONDITION_DIM_WITH_TASK:
        raise ValueError(f"expected x[:,{CONDITION_DIM_WITH_TASK}], got {x_all.shape}")
    if abnormal.sum() == 0:
        raise ValueError("dataset has no active abnormal rows")
    x, y, episodes = x_all[abnormal], y_all[abnormal], data["episode_id"][abnormal]
    train, validation = episode_split(episodes, 0.2, args.seed)
    if not train.any() or not validation.any():
        raise ValueError("need at least two abnormal episodes for episode-level validation")
    x_mean, x_scale = x[train].mean(0), x[train].std(0); x_scale[x_scale < 1e-6] = 1.0
    y_mean, y_scale = y[train].mean(0), y[train].std(0); y_scale[y_scale < 1e-6] = 1.0
    tx = torch.tensor((x[train] - x_mean) / x_scale, dtype=torch.float32)
    ty = torch.tensor((y[train] - y_mean) / y_scale, dtype=torch.float32)
    vx = torch.tensor((x[validation] - x_mean) / x_scale, dtype=torch.float32)
    vy = torch.tensor((y[validation] - y_mean) / y_scale, dtype=torch.float32)
    torch.manual_seed(args.seed); torch.set_num_threads(4)
    model = AbnormalMixtureExpert(x.shape[1], args.components)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=2e-4)
    best = None
    for epoch in range(args.epochs):
        model.train(); optimizer.zero_grad()
        loss = mixture_nll(*model(tx), ty); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            model.eval()
            with torch.no_grad(): value = float(mixture_nll(*model(vx), vy))
            if best is None or value < best[0]: best = (value, epoch, copy.deepcopy(model.state_dict()))
    # Hyperparameter selection is complete. Refit from scratch on every allowed
    # abnormal row for exactly the selected number of updates; validation data
    # never enters threshold calibration or model selection again.
    selected_updates = int(best[1]) + 1
    x_mean, x_scale = x.mean(0), x.std(0); x_scale[x_scale < 1e-6] = 1.0
    y_mean, y_scale = y.mean(0), y.std(0); y_scale[y_scale < 1e-6] = 1.0
    tx_all = torch.tensor((x - x_mean) / x_scale, dtype=torch.float32)
    ty_all = torch.tensor((y - y_mean) / y_scale, dtype=torch.float32)
    torch.manual_seed(args.seed)
    model = AbnormalMixtureExpert(x.shape[1], args.components)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=2e-4)
    for _ in range(selected_updates):
        model.train(); optimizer.zero_grad(); final_loss = mixture_nll(*model(tx_all), ty_all)
        final_loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
    model.eval()
    payload = {
        "model_type": "conditional_diagonal_gaussian_mixture",
        "state_dict": model.state_dict(), "input_dim": x.shape[1], "components": args.components,
        "x_mean": x_mean, "x_scale": x_scale, "y_mean": y_mean, "y_scale": y_scale,
        "feature_schema_sha256": schema_sha256(), "training_seed": args.seed,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True); torch.save(payload, args.out)
    digest = hashlib.sha256(args.out.read_bytes()).hexdigest()
    report = {
        "status": "trained_not_calibrated", "normal_rows_used": 0,
        "abnormal_rows": int(len(x)), "train_rows": int(train.sum()),
        "validation_rows": int(validation.sum()), "validation_episodes": int(len(np.unique(episodes[validation]))),
        "best_validation_nll_standardized": best[0], "best_epoch": best[1],
        "final_refit_rows": int(len(x)), "final_refit_updates": selected_updates,
        "final_refit_nll_standardized": float(final_loss),
        "components": args.components, "checkpoint_sha256": digest,
        "warning": "Do not inspect held-out test data before fusion calibration is frozen.",
    }
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
