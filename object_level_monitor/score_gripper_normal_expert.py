"""Score gripper responses under a frozen normal ensemble."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from train_gripper_normal_expert import Expert


LOG_2PI = float(np.log(2.0 * np.pi))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    data = np.load(args.dataset, allow_pickle=False)
    x, y = data["x"].astype(np.float64), data["y"].astype(np.float64)
    checkpoint = torch.load(args.model, map_location="cpu", weights_only=False)
    member_logp = []
    torch.set_num_threads(2)
    for saved in checkpoint["members"]:
        net = Expert(int(checkpoint["input_dim"]))
        net.load_state_dict(saved["state_dict"])
        net.eval()
        x_mean, x_scale = np.asarray(saved["x_mean"]), np.asarray(saved["x_scale"])
        y_mean, y_scale = np.asarray(saved["y_mean"]), np.asarray(saved["y_scale"])
        with torch.no_grad():
            raw = net(torch.tensor((x - x_mean) / x_scale, dtype=torch.float32)).numpy()
        mean = raw[:, :2] * y_scale + y_mean
        variance = np.exp(np.clip(raw[:, 2:], -6.0, 3.0)) * y_scale**2
        variance *= np.asarray(checkpoint["variance_scale_global"])
        variance = np.maximum(variance, 1e-12)
        member_logp.append(
            -0.5 * (np.log(variance) + (y - mean) ** 2 / variance + LOG_2PI).sum(1)
        )
    values = np.asarray(member_logp)
    maximum = values.max(0)
    logp = maximum + np.log(np.exp(values - maximum).mean(0))
    np.savez_compressed(
        args.out,
        normal_logp=logp,
        abnormal=data["abnormal"],
        episode_id=data["episode_id"],
        action_index=data["action_index"],
        source_id=data["source_id"],
    )
    print({"rows": len(x), "finite": bool(np.isfinite(logp).all())})


if __name__ == "__main__":
    main()
