"""Zero-fit ensemble scoring restricted to explicitly selected episodes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from conditional_relational_dual_expert import conditional_features, make_model
from libero_droid_transfer import load_trace


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, action="append", required=True)
    parser.add_argument("--trace", type=Path, action="append", required=True)
    parser.add_argument("--trace-map", type=Path, required=True)
    parser.add_argument("--private-map", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.model) < 3:
        raise ValueError("at least three models required")
    trace_rows = json.loads(args.trace_map.read_text(encoding="utf-8"))["files"]
    if len(trace_rows) != len(args.trace):
        raise ValueError("trace-map order must match repeated --trace arguments")
    private = json.loads(args.private_map.read_text(encoding="utf-8"))["records"]
    selected = {}
    for row in private:
        selected.setdefault((row["suite"], int(row["task_id"])), set()).add(int(row["episode_idx"]))

    import torch
    from torch import nn
    members = []
    for path in args.model:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        config = checkpoint["config"]
        model = make_model(torch, nn, config["input_dim"], config["output_dim"], config["seed"])
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        members.append((checkpoint, model))

    keys = ("normal_logp", "abnormal_logp", "ensemble_std", "abnormal",
            "task_id", "episode_idx", "action_index", "file_index")
    arrays = {key: [] for key in keys}
    for file_index, (path, mapping) in enumerate(zip(args.trace, trace_rows)):
        data, meta = load_trace(path)
        episodes = np.asarray(meta["episode_idx"], np.int32)
        x, y = conditional_features(data, episodes, 20, gripper_scale=1.0)
        keep = np.isin(episodes, sorted(selected[(mapping["suite"], int(mapping["task_id"]))]))
        x, y = x[keep], y[keep]
        normal_scores, abnormal_scores = [], []
        for checkpoint, model in members:
            X = torch.from_numpy(((x - checkpoint["xcenter"]) / checkpoint["xscale"]).astype(np.float32))
            Y = torch.from_numpy(((y - checkpoint["ycenter"]) / checkpoint["yscale"]).astype(np.float32))
            with torch.no_grad():
                normal_scores.append(model.logp(X, Y, False).numpy())
                abnormal_scores.append(model.logp(X, Y, True).numpy())
        normal_scores = np.stack(normal_scores)
        abnormal_scores = np.stack(abnormal_scores)
        arrays["normal_logp"].append(normal_scores.mean(0))
        arrays["abnormal_logp"].append(abnormal_scores.mean(0))
        arrays["ensemble_std"].append((abnormal_scores - normal_scores).std(0))
        for key in ("abnormal", "task_id", "episode_idx", "action_index"):
            arrays[key].append(np.asarray(meta[key])[keep])
        arrays["file_index"].append(np.full(int(keep.sum()), file_index, np.int32))

    payload = {key: np.concatenate(values) for key, values in arrays.items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **payload)
    print(json.dumps({"models": len(members), "files": len(args.trace),
                      "selected_episodes": len(private), "rows": len(payload["normal_logp"])}, indent=2))


if __name__ == "__main__":
    main()
