#!/usr/bin/env python3
"""Score target traces with a frozen source-trained dual-expert ensemble."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

from libero_droid_transfer import load_trace
from relational_dual_expert import make_model, relational_features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, action="append", required=True)
    parser.add_argument("--trace", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.model) < 3:
        raise ValueError("at least three ensemble members are required")
    import torch
    from torch import nn
    models = []
    for path in args.model:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        cfg = checkpoint["config"]
        dummy = np.zeros((cfg["components"], checkpoint["summary"]["feature_dim"]), np.float32)
        model = make_model(torch, nn, cfg["components"], checkpoint["summary"]["feature_dim"], cfg["seed"], dummy, dummy)
        model.load_state_dict(checkpoint["state_dict"]); model.eval()
        models.append((checkpoint, model))
    feature_set = models[0][0]["config"]["feature_set"]
    if any(x[0]["config"]["feature_set"] != feature_set for x in models):
        raise ValueError("ensemble feature sets differ")
    arrays = {key: [] for key in ("normal_logp", "abnormal_logp", "ensemble_std", "abnormal", "task_id", "episode_idx", "action_index", "file_index")}
    for file_index, path in enumerate(args.trace):
        data, meta = load_trace(path)
        raw = relational_features(data["command_history_cartesian_velocity"], data["response_eef_delta_xyz_euler"], 20, feature_set)
        normal, abnormal = [], []
        for checkpoint, model in models:
            x = torch.from_numpy(((raw - checkpoint["center"]) / checkpoint["scale"]).astype(np.float32))
            with torch.no_grad(): normal.append(model.logp(x, False).numpy()); abnormal.append(model.logp(x, True).numpy())
        normal, abnormal = np.stack(normal), np.stack(abnormal)
        arrays["normal_logp"].append(normal.mean(0)); arrays["abnormal_logp"].append(abnormal.mean(0))
        arrays["ensemble_std"].append((abnormal - normal).std(0))
        for key in ("abnormal", "task_id", "episode_idx", "action_index"): arrays[key].append(meta[key])
        arrays["file_index"].append(np.full(len(raw), file_index, np.int32))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **{key: np.concatenate(value) for key, value in arrays.items()})
    print(json.dumps({"models": len(models), "files": len(args.trace), "rows": int(sum(map(len, arrays["normal_logp"])))}, indent=2))

if __name__ == "__main__": main()
