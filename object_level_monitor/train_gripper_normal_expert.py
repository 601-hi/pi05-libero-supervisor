"""Train a task-ID-free heteroscedastic normal gripper-response ensemble."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


class Expert(torch.nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 128),
            torch.nn.SiLU(),
            torch.nn.Linear(128, 128),
            torch.nn.SiLU(),
            torch.nn.Linear(128, 4),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def nll(raw: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mean, logvar = raw[:, :2], raw[:, 2:].clamp(-6.0, 3.0)
    return 0.5 * (logvar + (target - mean).square() * torch.exp(-logvar)).sum(1).mean()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    data = np.load(args.dataset, allow_pickle=False)
    x = data["x"].astype(np.float64)
    y = data["y"].astype(np.float64)
    episode_id = data["episode_id"]
    if data["abnormal"].any():
        raise ValueError("normal training dataset contains applied gripper disturbances")
    if x.shape[1] != 58 or y.shape[1] != 2:
        raise ValueError(f"unexpected shapes x={x.shape}, y={y.shape}")

    fold_id = np.asarray(
        [int(hashlib.sha256(str(e).encode()).hexdigest()[:8], 16) % args.folds for e in episode_id]
    )
    members, summaries = [], []
    oof_mean, oof_var = np.zeros_like(y), np.zeros_like(y)
    torch.set_num_threads(2)
    for fold in range(args.folds):
        train, valid = fold_id != fold, fold_id == fold
        x_mean, x_scale = x[train].mean(0), x[train].std(0)
        y_mean, y_scale = y[train].mean(0), y[train].std(0)
        x_scale[x_scale < 1e-6] = 1.0
        y_scale[y_scale < 1e-8] = 1.0
        tx = torch.tensor((x[train] - x_mean) / x_scale, dtype=torch.float32)
        ty = torch.tensor((y[train] - y_mean) / y_scale, dtype=torch.float32)
        vx = torch.tensor((x[valid] - x_mean) / x_scale, dtype=torch.float32)
        vy = torch.tensor((y[valid] - y_mean) / y_scale, dtype=torch.float32)
        seed = 20260970 + fold
        torch.manual_seed(seed)
        net, best = Expert(x.shape[1]), None
        optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=2e-4)
        for epoch in range(args.epochs):
            net.train()
            optimizer.zero_grad()
            loss = nll(net(tx), ty)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            optimizer.step()
            if epoch % 10 == 0 or epoch == args.epochs - 1:
                net.eval()
                with torch.no_grad():
                    validation_nll = float(nll(net(vx), vy))
                if best is None or validation_nll < best[0]:
                    best = (validation_nll, epoch, copy.deepcopy(net.state_dict()))
        assert best is not None
        net.load_state_dict(best[2])
        net.eval()
        with torch.no_grad():
            raw = net(vx).numpy()
        oof_mean[valid] = raw[:, :2] * y_scale + y_mean
        oof_var[valid] = np.exp(np.clip(raw[:, 2:], -6.0, 3.0)) * y_scale**2
        members.append(
            {
                "state_dict": net.state_dict(),
                "x_mean": x_mean,
                "x_scale": x_scale,
                "y_mean": y_mean,
                "y_scale": y_scale,
                "seed": seed,
            }
        )
        summaries.append(
            {
                "fold": fold,
                "validation_nll": best[0],
                "best_epoch": best[1],
                "train_rows": int(train.sum()),
                "validation_rows": int(valid.sum()),
            }
        )

    variance_scale = np.clip(
        np.mean((y - oof_mean) ** 2 / np.maximum(oof_var, 1e-12), axis=0), 0.25, 25.0
    )
    checkpoint = {
        "model_type": "task_agnostic_normal_gripper_response_ensemble",
        "input_dim": x.shape[1],
        "output_dim": 2,
        "members": members,
        "variance_scale_global": variance_scale,
        "task_identity_used": False,
        "folds": args.folds,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.out)
    digest = hashlib.sha256(args.out.read_bytes()).hexdigest()
    report = {
        "status": "trained_not_calibrated",
        "rows": len(x),
        "episodes": len(np.unique(episode_id)),
        "folds": summaries,
        "variance_scale_global": variance_scale.tolist(),
        "checkpoint_sha256": digest,
    }
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
