"""Train a frozen-feature visual head on causally aligned ambiguous chunks."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


class Head(torch.nn.Module):
    def __init__(self, dim: int, kind: str):
        super().__init__()
        self.net = (
            torch.nn.Linear(dim, 1)
            if kind == "logistic"
            else torch.nn.Sequential(
                torch.nn.Linear(dim, 64), torch.nn.SiLU(),
                torch.nn.Linear(64, 32), torch.nn.SiLU(),
                torch.nn.Linear(32, 1),
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def projection(input_dim: int, output_dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((input_dim, output_dim)).astype(np.float32) / np.sqrt(output_dim))


def features(data, projection_dim: int, seed: int) -> np.ndarray:
    groups = ["base_current", "base_delta", "wrist_current", "wrist_delta"]
    input_dim = int(np.prod(data[groups[0]].shape[1:]))
    matrix = projection(input_dim, projection_dim, seed)
    projected = [np.asarray(data[name], dtype=np.float32).reshape(len(data[name]), -1) @ matrix for name in groups]
    # Condition features are deployable command/state context, while the two
    # log-likelihoods summarize the frozen dynamics experts for the completed chunk.
    columns = projected + [
        np.asarray(data["condition_mean"], dtype=np.float32),
        np.asarray(data["condition_last"], dtype=np.float32),
        np.asarray(data["normal_logp_mean"], dtype=np.float32)[:, None],
        np.asarray(data["abnormal_logp_mean"], dtype=np.float32)[:, None],
    ]
    return np.concatenate(columns, axis=1).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--features", type=Path, default=None, help="Optional precomputed feature matrix NPZ")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--kind", choices=["logistic", "mlp_2layer"], default="logistic")
    parser.add_argument("--projection-dim", type=int, default=64)
    parser.add_argument("--margin", type=float, default=0.5)
    parser.add_argument("--margin-weight", type=float, default=0.25)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260904)
    args = parser.parse_args()

    data = np.load(args.dataset, allow_pickle=False)
    clean = np.isclose(data["scale"], 1.0)
    active = data["previous_active_any"].astype(bool)
    ambiguous = data["previous_ambiguous_any"].astype(bool)
    selected = ambiguous & (clean | active)
    labels = active[selected].astype(np.float32)
    if labels.sum() == 0 or labels.sum() == len(labels):
        raise ValueError("selected training set must contain both clean and active chunks")

    if args.features is None:
        all_features = features(data, args.projection_dim, args.seed)
    else:
        cached = np.load(args.features, allow_pickle=False)
        all_features = cached["features"].astype(np.float32)
        if len(all_features) != len(data["episode_id"]):
            raise ValueError("feature cache row mismatch")
        if int(cached["projection_dim"]) != args.projection_dim or int(cached["projection_seed"]) != args.seed:
            raise ValueError("feature cache projection configuration mismatch")
    x = all_features[selected]
    mean, scale = x.mean(0), x.std(0)
    scale[scale < 1e-6] = 1.0
    x = (x - mean) / scale

    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    model = Head(x.shape[1], args.kind)
    tx, ty = torch.tensor(x), torch.tensor(labels)
    pos, neg = float(labels.sum()), float(len(labels) - labels.sum())
    bce = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(neg / max(pos, 1.0)))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    for _ in range(args.epochs):
        optimizer.zero_grad()
        logits = model(tx)
        signed = (2.0 * ty - 1.0) * logits
        loss = bce(logits, ty) + args.margin_weight * torch.relu(args.margin - signed).mean()
        loss.backward()
        optimizer.step()

    payload = {
        "state_dict": model.state_dict(), "kind": args.kind, "input_dim": x.shape[1],
        "projection_input_dim": int(np.prod(data["base_current"].shape[1:])),
        "projection_dim": args.projection_dim, "projection_seed": args.seed,
        "mean": mean, "scale": scale, "margin": args.margin,
        "margin_weight": args.margin_weight, "epochs": args.epochs,
        "causal_alignment": "visual transition at chunk k scores executed chunk k-1",
    }
    torch.save(payload, args.out)
    with torch.no_grad():
        probability = torch.sigmoid(model(tx)).numpy()
    report = {
        "selected_chunks": int(len(labels)),
        "active_chunks": int(labels.sum()),
        "clean_chunks": int((labels == 0).sum()),
        "train_loss": float(loss),
        "active_probability_median": float(np.median(probability[labels == 1])),
        "clean_probability_median": float(np.median(probability[labels == 0])),
        "checkpoint_sha256": hashlib.sha256(args.out.read_bytes()).hexdigest(),
        "warning": "Training fit only. Threshold and model selection belong exclusively to the calibration split.",
    }
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
