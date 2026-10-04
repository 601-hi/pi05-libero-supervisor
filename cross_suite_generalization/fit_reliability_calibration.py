#!/usr/bin/env python3
"""Fit source-only reliability references for a frozen dual-expert ensemble."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from relational_dual_expert import make_model, relational_features


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_model(path, torch, nn):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    cfg = checkpoint["config"]
    dummy = np.zeros((cfg["components"], checkpoint["summary"]["feature_dim"]), np.float32)
    model = make_model(torch, nn, cfg["components"], checkpoint["summary"]["feature_dim"], cfg["seed"], dummy, dummy)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return checkpoint, model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-directory", type=Path, required=True)
    parser.add_argument("--model", type=Path, action="append", required=True, help="Bootstrap ensemble member; repeat at least 3 times")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sequence-scores-output", type=Path, required=True)
    parser.add_argument("--normal-lr-quantile", type=float, default=0.90)
    parser.add_argument("--abnormal-lr-quantile", type=float, default=0.10)
    args = parser.parse_args()
    if len(args.model) < 3:
        raise ValueError("at least three independently trained/bootstrap models are required")
    import torch
    from torch import nn
    manifest = json.loads((args.data_directory / "manifest.json").read_text(encoding="utf-8"))
    cal_path = Path(manifest["output_files"]["calibration"]["path"])
    if not cal_path.is_absolute():
        cal_path = args.data_directory / cal_path
    data = np.load(cal_path, allow_pickle=False)
    counts = [int(row["exported_transitions"]) for row in manifest["episodes"] if row["split"] == "calibration" and row["accepted"]]
    if sum(counts) != len(data["response_eef_delta_xyz_euler"]):
        raise ValueError("calibration episode counts do not match transition array")
    episode = np.concatenate([np.full(count, index, np.int32) for index, count in enumerate(counts)])
    action = np.concatenate([np.arange(count, dtype=np.int32) for count in counts])
    reliability_fit = episode % 2 == 0
    sequence_tune = ~reliability_fit
    checkpoints_models = [load_model(path, torch, nn) for path in args.model]
    first = checkpoints_models[0][0]
    cfg = first["config"]
    raw_normal = relational_features(data["command_history_cartesian_velocity"], data["response_eef_delta_xyz_euler"], 15, cfg["feature_set"])
    raw_abnormal = np.concatenate([
        relational_features(data["command_history_cartesian_velocity"], data["response_eef_delta_xyz_euler"] * scale, 15, cfg["feature_set"])
        for scale in cfg["scales"]
    ])

    def score(raw):
        lrs, bests = [], []
        for checkpoint, model in checkpoints_models:
            if checkpoint["config"]["feature_set"] != cfg["feature_set"]:
                raise ValueError("ensemble feature sets differ")
            x = torch.from_numpy(((raw - checkpoint["center"]) / checkpoint["scale"]).astype(np.float32))
            with torch.no_grad():
                n, a = model.logp(x, False).numpy(), model.logp(x, True).numpy()
            lrs.append(a - n)
            bests.append(np.maximum(a, n))
        return np.stack(lrs), np.stack(bests)

    normal_lr, normal_best = score(raw_normal)
    abnormal_lr, abnormal_best = score(raw_abnormal)
    abnormal_fit = np.tile(reliability_fit, len(cfg["scales"]))
    abnormal_tune = np.tile(sequence_tune, len(cfg["scales"]))
    pooled_best = np.concatenate([normal_best.mean(0)[reliability_fit], abnormal_best.mean(0)[abnormal_fit]])
    pooled_std = np.concatenate([normal_lr.std(0)[reliability_fit], abnormal_lr.std(0)[abnormal_fit]])
    normal_mean_lr, abnormal_mean_lr = normal_lr.mean(0), abnormal_lr.mean(0)
    payload = {
        "schema_version": 1,
        "source_only": True,
        "calibration_data_sha256": sha256(cal_path),
        "model_sha256": [sha256(path) for path in args.model],
        "models": [str(path) for path in args.model],
        "best_logp_reference": pooled_best.tolist(),
        "ensemble_std_reference": pooled_std.tolist(),
        "normal_lr_threshold": float(np.quantile(normal_mean_lr[reliability_fit], args.normal_lr_quantile)),
        "abnormal_lr_threshold": float(np.quantile(abnormal_mean_lr[abnormal_fit], args.abnormal_lr_quantile)),
        "quantiles": {"normal": args.normal_lr_quantile, "abnormal": args.abnormal_lr_quantile},
        "overlap_warning": bool(np.quantile(normal_mean_lr[reliability_fit], args.normal_lr_quantile) >= np.quantile(abnormal_mean_lr[abnormal_fit], args.abnormal_lr_quantile)),
        "split_rule": "calibration episode index even=reliability fit; odd=sequence tune",
        "counts": {"normal_total": len(raw_normal), "synthetic_abnormal_total": len(raw_abnormal),
                   "reliability_fit_normal": int(reliability_fit.sum()), "sequence_tune_normal": int(sequence_tune.sum())},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    normal_n = normal_best.mean(0); normal_a = normal_n + normal_mean_lr
    abnormal_n = abnormal_best.mean(0) - np.maximum(abnormal_mean_lr, 0)
    abnormal_a = abnormal_n + abnormal_mean_lr
    # Preserve likelihood-ratio exactly; absolute N/A decomposition for synthetic
    # samples comes from the ensemble means below rather than the approximations.
    def ensemble_logps(raw):
        ns, ass = [], []
        for checkpoint, model in checkpoints_models:
            x = torch.from_numpy(((raw - checkpoint["center"]) / checkpoint["scale"]).astype(np.float32))
            with torch.no_grad(): ns.append(model.logp(x, False).numpy()); ass.append(model.logp(x, True).numpy())
        return np.stack(ns).mean(0), np.stack(ass).mean(0)
    normal_n, normal_a = ensemble_logps(raw_normal)
    abnormal_n, abnormal_a = ensemble_logps(raw_abnormal)
    repeated_episode = np.concatenate([episode + (scale_index + 1) * len(counts) for scale_index in range(len(cfg["scales"]))])
    repeated_action = np.tile(action, len(cfg["scales"]))
    args.sequence_scores_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.sequence_scores_output,
        normal_logp=np.concatenate([normal_n[sequence_tune], abnormal_n[abnormal_tune]]),
        abnormal_logp=np.concatenate([normal_a[sequence_tune], abnormal_a[abnormal_tune]]),
        ensemble_std=np.concatenate([normal_lr.std(0)[sequence_tune], abnormal_lr.std(0)[abnormal_tune]]),
        abnormal=np.concatenate([np.zeros(sequence_tune.sum(), bool), np.ones(abnormal_tune.sum(), bool)]),
        episode=np.concatenate([episode[sequence_tune], repeated_episode[abnormal_tune]]),
        action=np.concatenate([action[sequence_tune], repeated_action[abnormal_tune]]),
    )
    print(json.dumps({k: payload[k] for k in ("source_only", "normal_lr_threshold", "abnormal_lr_threshold", "overlap_warning", "counts")}, indent=2))


if __name__ == "__main__":
    main()
