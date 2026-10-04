#!/usr/bin/env python3
"""Train a domain-normalized heteroscedastic response encoder; validation is default."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np


INPUT_KEYS = (
    "command_history_cartesian_velocity",
    "state_eef_pose_xyz_euler",
    "state_joint_position",
    "state_gripper_position",
)
TARGET_KEYS = (
    "response_eef_delta_xyz_euler",
    "response_joint_delta",
)
GRIPPER_TARGET_KEY = "response_gripper_delta"


def flatten_join(data, keys):
    return np.concatenate([data[key].reshape(len(data[key]), -1) for key in keys], axis=1)


def normalized(data, keys, normalization):
    pieces = []
    for key in keys:
        value = data[key].reshape(len(data[key]), -1).astype(np.float32)
        center = np.asarray(normalization[key]["median"], dtype=np.float32)
        scale = np.asarray(normalization[key]["iqr_scale"], dtype=np.float32)
        pieces.append((value - center) / scale)
    return np.concatenate(pieces, axis=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--readiness", type=Path, required=True)
    parser.add_argument("--data-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train", action="store_true", help="Actually update weights; default is validation only")
    args = parser.parse_args()
    import torch
    from torch import nn

    config = json.loads(args.config.read_text(encoding="utf-8"))
    readiness = json.loads(args.readiness.read_text(encoding="utf-8"))
    if not readiness.get("authorized", False):
        raise RuntimeError(f"training bundle is not authorized: {readiness.get('violations')}")
    torch.set_num_threads(int(config.get("torch_num_threads", 1)))
    manifest = json.loads((args.data_directory / "manifest.json").read_text(encoding="utf-8"))
    train = np.load(args.data_directory / manifest["output_files"]["train"]["path"], allow_pickle=False)
    calibration = np.load(
        args.data_directory / manifest["output_files"]["calibration"]["path"], allow_pickle=False
    )
    x_train = normalized(train, INPUT_KEYS, readiness["normalization"])
    y_train = normalized(train, TARGET_KEYS, readiness["normalization"])
    x_cal = normalized(calibration, INPUT_KEYS, readiness["normalization"])
    y_cal = normalized(calibration, TARGET_KEYS, readiness["normalization"])
    gripper_train_raw = train[GRIPPER_TARGET_KEY].reshape(len(train[GRIPPER_TARGET_KEY]), -1).astype(np.float32)
    gripper_cal_raw = calibration[GRIPPER_TARGET_KEY].reshape(len(calibration[GRIPPER_TARGET_KEY]), -1).astype(np.float32)
    gripper_active_threshold = float(config.get("gripper_active_threshold", 1e-6))
    gripper_train_active = (np.abs(gripper_train_raw) > gripper_active_threshold).astype(np.float32)
    gripper_cal_active = (np.abs(gripper_cal_raw) > gripper_active_threshold).astype(np.float32)
    active_values = gripper_train_raw[gripper_train_active[:, 0] > 0.5, 0]
    if len(active_values) == 0:
        raise RuntimeError("training split contains no active gripper transitions")
    gripper_active_center = float(np.median(active_values))
    gripper_active_scale = float(np.percentile(active_values, 75) - np.percentile(active_values, 25))
    if gripper_active_scale <= 1e-8:
        gripper_active_scale = float(np.std(active_values))
    if gripper_active_scale <= 1e-8:
        raise RuntimeError("active gripper magnitude has zero robust scale")
    gripper_train_value = (gripper_train_raw - gripper_active_center) / gripper_active_scale
    gripper_cal_value = (gripper_cal_raw - gripper_active_center) / gripper_active_scale
    torch.manual_seed(config["seed"])
    model = nn.Sequential(
        nn.Linear(x_train.shape[1], config["hidden_dim"]), nn.SiLU(),
        nn.Linear(config["hidden_dim"], config["hidden_dim"]), nn.SiLU(),
        # 2 * 13 continuous response dimensions, one gripper-event logit,
        # and mean/log-variance for gripper magnitude conditional on activity.
        nn.Linear(config["hidden_dim"], 2 * y_train.shape[1] + 3),
    )
    with torch.no_grad():
        probe = model(torch.from_numpy(x_train[: min(8, len(x_train))]))
    summary = {
        "mode": "train" if args.train else "validate_only",
        "train_samples": len(x_train), "calibration_samples": len(x_cal),
        "input_dim": x_train.shape[1], "continuous_target_dim": y_train.shape[1],
        "gripper_active_fraction_train": float(gripper_train_active.mean()),
        "gripper_active_fraction_calibration": float(gripper_cal_active.mean()),
        "gripper_active_center": gripper_active_center,
        "gripper_active_scale": gripper_active_scale,
        "probe_output_shape": list(probe.shape), "finite_probe": bool(torch.isfinite(probe).all()),
    }
    if not args.train:
        print(json.dumps(summary, indent=2))
        return

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
    x_tensor, y_tensor = torch.from_numpy(x_train), torch.from_numpy(y_train)
    x_cal_tensor, y_cal_tensor = torch.from_numpy(x_cal), torch.from_numpy(y_cal)
    grip_active_tensor = torch.from_numpy(gripper_train_active)
    grip_value_tensor = torch.from_numpy(gripper_train_value)
    grip_cal_active_tensor = torch.from_numpy(gripper_cal_active)
    grip_cal_value_tensor = torch.from_numpy(gripper_cal_value)
    generator = torch.Generator().manual_seed(config["seed"])
    history = []
    best_calibration = float("inf")
    best_epoch = -1
    best_state = None
    stale_epochs = 0
    patience = int(config.get("early_stopping_patience", 8))
    minimum_delta = float(config.get("early_stopping_min_delta", 1e-4))

    def gaussian_nll(prediction, target):
        mean, log_variance = prediction.chunk(2, dim=1)
        log_variance = log_variance.clamp(-8.0, 6.0)
        return 0.5 * (log_variance + (target - mean) ** 2 * torch.exp(-log_variance))

    positive_count = float(gripper_train_active.sum())
    negative_count = float(len(gripper_train_active) - positive_count)
    gripper_positive_weight = torch.tensor([negative_count / positive_count], dtype=torch.float32)
    event_loss_fn = nn.BCEWithLogitsLoss(pos_weight=gripper_positive_weight)

    def structured_loss(prediction, target, grip_active, grip_value):
        continuous_prediction = prediction[:, : 2 * target.shape[1]]
        event_logit = prediction[:, 2 * target.shape[1] : 2 * target.shape[1] + 1]
        magnitude_prediction = prediction[:, 2 * target.shape[1] + 1 :]
        continuous_loss = gaussian_nll(continuous_prediction, target).mean()
        event_loss = event_loss_fn(event_logit, grip_active)
        active_mask = grip_active[:, 0] > 0.5
        if bool(active_mask.any()):
            magnitude_loss = gaussian_nll(magnitude_prediction[active_mask], grip_value[active_mask]).mean()
        else:
            magnitude_loss = prediction.sum() * 0.0
        return continuous_loss + event_loss + magnitude_loss, continuous_loss, event_loss, magnitude_loss

    for epoch in range(config["epochs"]):
        model.train()
        train_loss_sum = 0.0
        train_items = 0
        for indices in torch.randperm(len(x_tensor), generator=generator).split(config["batch_size"]):
            prediction = model(x_tensor[indices])
            loss, _, _, _ = structured_loss(
                prediction, y_tensor[indices], grip_active_tensor[indices], grip_value_tensor[indices]
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss_sum += float(loss.detach()) * len(indices)
            train_items += len(indices)
        model.eval()
        calibration_sum = 0.0
        calibration_component_sum = np.zeros(3, dtype=np.float64)
        calibration_items = 0
        with torch.no_grad():
            for start in range(0, len(x_cal_tensor), config["batch_size"]):
                stop = start + config["batch_size"]
                loss, cont_loss, event_loss, magnitude_loss = structured_loss(
                    model(x_cal_tensor[start:stop]), y_cal_tensor[start:stop],
                    grip_cal_active_tensor[start:stop], grip_cal_value_tensor[start:stop],
                )
                count = len(x_cal_tensor[start:stop])
                calibration_sum += float(loss) * count
                calibration_component_sum += np.asarray(
                    [float(cont_loss), float(event_loss), float(magnitude_loss)]
                ) * count
                calibration_items += count
        train_nll = train_loss_sum / train_items
        calibration_nll = calibration_sum / calibration_items
        component_means = calibration_component_sum / calibration_items
        record = {
            "epoch": epoch + 1, "train_loss": train_nll, "calibration_loss": calibration_nll,
            "calibration_continuous_nll": float(component_means[0]),
            "calibration_gripper_event_bce": float(component_means[1]),
            "calibration_gripper_active_nll": float(component_means[2]),
        }
        history.append(record)
        print(json.dumps(record), flush=True)
        if calibration_nll < best_calibration - minimum_delta:
            best_calibration = calibration_nll
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break
    if best_state is None:
        raise RuntimeError("training produced no finite best checkpoint")
    model.load_state_dict(best_state)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary.update({
        "epochs_completed": len(history), "best_epoch": best_epoch,
        "best_calibration_loss": best_calibration,
        "readiness_sha256": hashlib.sha256(args.readiness.read_bytes()).hexdigest(),
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
    })
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    torch.save({"state_dict": model.state_dict(), "config": config, "summary": summary}, temporary)
    temporary.replace(args.output)
    history_path = args.output.with_suffix(".history.json")
    history_path.write_text(json.dumps({"summary": summary, "epochs": history}, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
