#!/usr/bin/env python3
"""Contrastive-margin fine-tuning for the known attenuation density expert.

The frozen normal expert supplies a fixed reference log density. The abnormal
expert retains its density NLL while being encouraged to rank active faults
above clean normal responses. Test data must never be used for selection.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from dynamics_feature_schema import schema_sha256
from score_dynamics_experts import normal_logp
from train_abnormal_dynamics_expert import AbnormalMixtureExpert, mixture_nll


def physical_logp(model, xz, yz, y_scale):
    logits, mean, logvar = model(xz)
    residual = yz[:, None, :] - mean
    component = -0.5 * (logvar + residual.square() * torch.exp(-logvar)).sum(-1)
    component -= 1.5 * np.log(2.0 * np.pi)
    standardized = torch.logsumexp(torch.log_softmax(logits, -1) + component, -1)
    return standardized - float(np.log(y_scale).sum())


def robust(values):
    center = float(np.median(values))
    scale = float(np.subtract(*np.percentile(values, [75, 25])) / 1.349)
    return center, max(scale, 1e-3)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--normal", type=Path, required=True)
    p.add_argument("--initial-abnormal", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--margin-abnormal", type=float, default=2.0)
    p.add_argument("--margin-normal", type=float, default=0.0)
    p.add_argument("--lambda-abnormal", type=float, default=0.25)
    p.add_argument("--lambda-normal", type=float, default=0.10)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--seed", type=int, default=20260904)
    a = p.parse_args()

    data = np.load(a.dataset, allow_pickle=False)
    x, y = data["x"].astype(np.float64), data["y"].astype(np.float64)
    active = data["abnormal"].astype(bool)
    clean = np.isclose(data["scale"], 1.0)
    task = data["task_id"].astype(int)
    if not active.any() or not clean.any():
        raise ValueError("requires both active abnormal and clean normal rows")
    normal_checkpoint = torch.load(a.normal, map_location="cpu")
    initial = torch.load(a.initial_abnormal, map_location="cpu")
    if initial.get("feature_schema_sha256") != schema_sha256():
        raise ValueError("feature schema mismatch")

    xm, xs = np.asarray(initial["x_mean"]), np.asarray(initial["x_scale"])
    ym, ys = np.asarray(initial["y_mean"]), np.asarray(initial["y_scale"])
    tx = torch.tensor((x - xm) / xs, dtype=torch.float32)
    ty = torch.tensor((y - ym) / ys, dtype=torch.float32)
    nlp = normal_logp(normal_checkpoint, x, y)
    tnlp = torch.tensor(nlp, dtype=torch.float32)

    torch.manual_seed(a.seed)
    torch.set_num_threads(4)
    model = AbnormalMixtureExpert(int(initial["input_dim"]), int(initial["components"]))
    model.load_state_dict(initial["state_dict"])
    model.eval()
    with torch.no_grad():
        initial_alp = physical_logp(model, tx, ty, ys).numpy()
    centers, scales = {}, {}
    for t in sorted(np.unique(task)):
        centers[t], scales[t] = robust((initial_alp - nlp)[clean & (task == t)])
    center_vec = torch.tensor([centers[int(t)] for t in task], dtype=torch.float32)
    scale_vec = torch.tensor([scales[int(t)] for t in task], dtype=torch.float32)
    active_t = torch.tensor(active)
    clean_t = torch.tensor(clean)

    optimizer = torch.optim.AdamW(model.parameters(), lr=a.learning_rate, weight_decay=2e-4)
    best = None
    history = []
    for epoch in range(a.epochs):
        model.train()
        optimizer.zero_grad()
        alp = physical_logp(model, tx, ty, ys)
        ratio_z = (alp - tnlp - center_vec) / scale_vec
        density = mixture_nll(*model(tx[active_t]), ty[active_t])
        abnormal_margin = torch.nn.functional.softplus(a.margin_abnormal - ratio_z[active_t]).mean()
        normal_margin = torch.nn.functional.softplus(a.margin_normal + ratio_z[clean_t]).mean()
        loss = density + a.lambda_abnormal * abnormal_margin + a.lambda_normal * normal_margin
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if epoch % 10 == 0 or epoch == a.epochs - 1:
            model.eval()
            with torch.no_grad():
                alp_eval = physical_logp(model, tx, ty, ys)
                rz_eval = (alp_eval - tnlp - center_vec) / scale_vec
                record = {
                    "epoch": epoch,
                    "loss": float(loss),
                    "density_nll_standardized": float(density),
                    "active_ratio_z_median": float(rz_eval[active_t].median()),
                    "clean_ratio_z_median": float(rz_eval[clean_t].median()),
                }
            history.append(record)
            # Select only by the declared training objective; calibration remains separate.
            if best is None or record["loss"] < best[0]:
                best = (record["loss"], epoch, copy.deepcopy(model.state_dict()))
    model.load_state_dict(best[2])
    payload = dict(initial)
    payload.update({
        "state_dict": model.state_dict(),
        "model_type": "conditional_diagonal_gaussian_mixture_with_contrastive_margin",
        "margin_training": {
            "margin_abnormal": a.margin_abnormal, "margin_normal": a.margin_normal,
            "lambda_abnormal": a.lambda_abnormal, "lambda_normal": a.lambda_normal,
            "epochs_requested": a.epochs, "selected_epoch": int(best[1]),
            "learning_rate": a.learning_rate, "seed": a.seed,
            "normal_reference_center_by_task": {str(int(k)): float(v) for k, v in centers.items()},
            "normal_reference_scale_by_task": {str(int(k)): float(v) for k, v in scales.items()},
        },
    })
    a.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, a.out)
    digest = hashlib.sha256(a.out.read_bytes()).hexdigest()
    report = {
        "status": "margin_finetuned_not_calibrated",
        "clean_rows": int(clean.sum()), "active_rows": int(active.sum()),
        "hyperparameters": payload["margin_training"],
        "best_training_objective": float(best[0]),
        "checkpoint_sha256": digest,
        "history": history,
        "warning": "Select fusion only on calibration data; do not tune from seed21.",
    }
    a.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "history"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
