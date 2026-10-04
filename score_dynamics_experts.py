#!/usr/bin/env python3
"""Score an exported dataset under frozen normal and abnormal experts.

This is deliberately separate from calibration: it produces raw comparable
log densities and never chooses an alarm threshold.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from dynamics_feature_schema import CONDITION_DIM_WITH_TASK, schema_sha256
from normal_dynamics_expert_runtime import _Expert
from train_abnormal_dynamics_expert import AbnormalMixtureExpert


LOG_2PI = float(np.log(2.0 * np.pi))


def normal_logp(checkpoint, x, y):
    member_logp = []
    for saved in checkpoint["members"]:
        net = _Expert(int(checkpoint["input_dim"])); net.load_state_dict(saved["state_dict"]); net.eval()
        xm = np.asarray(saved["x_mean"]); xs = np.asarray(saved["x_scale"])
        ym = np.asarray(saved["y_mean"]); ys = np.asarray(saved["y_scale"])
        with torch.no_grad(): raw = net(torch.tensor((x-xm)/xs, dtype=torch.float32)).numpy()
        mean = raw[:, :3] * ys + ym
        variance = np.exp(np.clip(raw[:, 3:], -6, 3)) * ys**2
        task = np.argmax(x[:, -10:], axis=1)
        scale = np.asarray([checkpoint["variance_scale"][int(t)] for t in task])
        variance = np.maximum(variance * scale, 1e-12)
        member_logp.append(-.5 * (np.log(variance) + (y-mean)**2/variance + LOG_2PI).sum(1))
    values = np.asarray(member_logp)
    maximum = values.max(0)
    return maximum + np.log(np.exp(values-maximum).mean(0))


def abnormal_logp(checkpoint, x, y):
    net = AbnormalMixtureExpert(int(checkpoint["input_dim"]), int(checkpoint["components"]))
    net.load_state_dict(checkpoint["state_dict"]); net.eval()
    xm, xs = np.asarray(checkpoint["x_mean"]), np.asarray(checkpoint["x_scale"])
    ym, ys = np.asarray(checkpoint["y_mean"]), np.asarray(checkpoint["y_scale"])
    with torch.no_grad(): logits, mean_z, logvar_z = net(torch.tensor((x-xm)/xs, dtype=torch.float32))
    logits, mean_z, logvar_z = logits.numpy(), mean_z.numpy(), logvar_z.numpy()
    target_z = (y-ym)/ys
    component = -.5 * (logvar_z + (target_z[:,None,:]-mean_z)**2/np.exp(logvar_z) + LOG_2PI).sum(2)
    logweight = logits - np.logaddexp.reduce(logits, axis=1)[:,None]
    joint = logweight + component
    maximum = joint.max(1)
    # Density transform from standardized y back to metres.
    return maximum + np.log(np.exp(joint-maximum[:,None]).sum(1)) - np.log(ys).sum()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--normal", type=Path, required=True)
    parser.add_argument("--abnormal-expert", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(); data = np.load(args.dataset, allow_pickle=False)
    x, y = data["x"].astype(np.float64), data["y"].astype(np.float64)
    if x.shape[1] != CONDITION_DIM_WITH_TASK: raise ValueError(f"bad feature dimension {x.shape}")
    normal = torch.load(args.normal, map_location="cpu")
    abnormal = torch.load(args.abnormal_expert, map_location="cpu")
    if abnormal.get("feature_schema_sha256") != schema_sha256(): raise ValueError("abnormal schema hash mismatch")
    nlp, alp = normal_logp(normal, x, y), abnormal_logp(abnormal, x, y)
    np.savez_compressed(args.out, normal_logp=nlp, abnormal_logp=alp,
                        abnormal=data["abnormal"], task_id=data["task_id"], episode_id=data["episode_id"],
                        action_index=data["action_index"], scale=data["scale"], onset_mode=data["onset_mode"])
    print({"rows": len(x), "normal_logp_finite": bool(np.isfinite(nlp).all()),
           "abnormal_logp_finite": bool(np.isfinite(alp).all())})


if __name__ == "__main__": main()
