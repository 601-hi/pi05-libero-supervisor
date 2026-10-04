#!/usr/bin/env python3
"""Map LIBERO traces into the frozen DROID response-encoder contract and score them.

Two tracks are deliberately separate:
  physical_zero_shot: controller specifications only, no LIBERO fit;
  normal_only_adaptation: robust marginals fit on explicitly supplied normal traces.
Task ids, reward, success, disturbance flags, and object state are never features.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


INPUT_KEYS = (
    "command_history_cartesian_velocity",
    "state_eef_pose_xyz_euler",
    "state_joint_position",
    "state_gripper_position",
)
TARGET_KEYS = ("response_eef_delta_xyz_euler", "response_joint_delta")
ALL_KEYS = INPUT_KEYS + TARGET_KEYS + ("response_gripper_delta",)


def quaternion_wxyz_to_euler(q):
    w, x, y, z = np.asarray(q, dtype=np.float64)
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return np.asarray([roll, pitch, yaw])


def wrapped_delta(after, before):
    return (np.asarray(after) - np.asarray(before) + np.pi) % (2 * np.pi) - np.pi


def gripper_open_fraction(qpos):
    # Panda two-finger separation, normalized by the nominal 0.08 m full opening.
    q = np.asarray(qpos, dtype=np.float64)
    return np.asarray([np.clip(abs(q[0] - q[1]) / 0.08, 0.0, 1.0)])


def intended_velocity(action, control_frequency_hz=20.0):
    action = np.asarray(action, dtype=np.float64)
    target_delta = np.concatenate((0.05 * np.clip(action[:3], -1, 1),
                                   0.5 * np.clip(action[3:6], -1, 1)))
    return target_delta * control_frequency_hz


def load_trace(path):
    episodes = defaultdict(list)
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("event") == "step":
                episodes[(int(row["task_id"]), int(row["episode_idx"]))].append(row)
    arrays = {key: [] for key in ALL_KEYS}
    metadata = {key: [] for key in ("task_id", "episode_idx", "action_index", "abnormal")}
    for (task_id, episode_idx), rows in sorted(episodes.items()):
        rows.sort(key=lambda row: int(row["action_index"]))
        indices = [int(row["action_index"]) for row in rows]
        if indices != list(range(len(rows))):
            raise ValueError(f"non-contiguous action indices in {path}: {(task_id, episode_idx)}")
        commands = np.asarray([intended_velocity(row["intended_action"]) for row in rows])
        for index in range(5, len(rows)):
            row = rows[index]
            before_euler = quaternion_wxyz_to_euler(row["eef_quat_before"])
            after_euler = quaternion_wxyz_to_euler(row["eef_quat_after"])
            before_grip = gripper_open_fraction(row["gripper_qpos_before"])
            after_grip = gripper_open_fraction(row["gripper_qpos_after"])
            arrays["command_history_cartesian_velocity"].append(commands[index - np.arange(6)])
            arrays["state_eef_pose_xyz_euler"].append(
                np.concatenate((np.asarray(row["eef_pos_before"]), before_euler))
            )
            arrays["state_joint_position"].append(row["joint_pos_before"])
            arrays["state_gripper_position"].append(before_grip)
            arrays["response_eef_delta_xyz_euler"].append(np.concatenate((
                np.asarray(row["eef_pos_after"]) - np.asarray(row["eef_pos_before"]),
                wrapped_delta(after_euler, before_euler),
            )))
            arrays["response_joint_delta"].append(
                np.asarray(row["joint_pos_after"]) - np.asarray(row["joint_pos_before"])
            )
            arrays["response_gripper_delta"].append(after_grip - before_grip)
            metadata["task_id"].append(task_id)
            metadata["episode_idx"].append(episode_idx)
            metadata["action_index"].append(index)
            metadata["abnormal"].append(bool(row.get("disturbance_active", False)))
    return ({key: np.asarray(value, dtype=np.float32) for key, value in arrays.items()},
            {key: np.asarray(value) for key, value in metadata.items()})


def concatenate_datasets(datasets):
    return {key: np.concatenate([dataset[key] for dataset in datasets], axis=0) for key in ALL_KEYS}


def robust_normalization(data):
    result = {}
    for key in ALL_KEYS:
        values = data[key].reshape(len(data[key]), -1).astype(np.float64)
        median = np.median(values, axis=0)
        scale = np.percentile(values, 75, axis=0) - np.percentile(values, 25, axis=0)
        # Continuous robot channels may contain nearly fixed dimensions. A train-only
        # 1%--99% span is a safer fallback than an arbitrary 1e-6 denominator.
        wide = 0.5 * (np.percentile(values, 99, axis=0) - np.percentile(values, 1, axis=0))
        scale = np.where(scale > 1e-8, scale, wide)
        scale = np.where(scale > 1e-8, scale, 1.0)
        result[key] = {"median": median.tolist(), "iqr_scale": scale.tolist()}
    return result


def normalize(data, keys, normalization):
    pieces = []
    for key in keys:
        value = data[key].reshape(len(data[key]), -1).astype(np.float32)
        center = np.asarray(normalization[key]["median"], dtype=np.float32)
        scale = np.asarray(normalization[key]["iqr_scale"], dtype=np.float32)
        pieces.append((value - center) / scale)
    return np.concatenate(pieces, axis=1)


def build_model(torch, nn, checkpoint, input_dim, target_dim):
    config = checkpoint["config"]
    model = nn.Sequential(
        nn.Linear(input_dim, config["hidden_dim"]), nn.SiLU(),
        nn.Linear(config["hidden_dim"], config["hidden_dim"]), nn.SiLU(),
        nn.Linear(config["hidden_dim"], 2 * target_dim + 3),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def continuous_score(model, torch, x, y):
    with torch.no_grad():
        prediction = model(torch.from_numpy(x)).numpy()
    dimension = y.shape[1]
    mean = prediction[:, :dimension]
    log_variance = np.clip(prediction[:, dimension:2 * dimension], -8.0, 6.0)
    per_dimension = 0.5 * (log_variance + (y - mean) ** 2 * np.exp(-log_variance))
    return per_dimension.mean(axis=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--droid-readiness", type=Path, required=True)
    parser.add_argument("--adapt-normal", type=Path, action="append", default=[])
    parser.add_argument("--evaluate", type=Path, action="append", required=True)
    parser.add_argument("--track", choices=("physical_zero_shot", "normal_only_adaptation"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import torch
    from torch import nn

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    torch.set_num_threads(1)
    droid_normalization = json.loads(args.droid_readiness.read_text(encoding="utf-8"))["normalization"]
    if args.track == "normal_only_adaptation":
        if not args.adapt_normal:
            raise ValueError("normal_only_adaptation requires --adapt-normal")
        fit_sets = [load_trace(path)[0] for path in args.adapt_normal]
        normalization = robust_normalization(concatenate_datasets(fit_sets))
    else:
        if args.adapt_normal:
            raise ValueError("physical_zero_shot forbids --adapt-normal")
        normalization = droid_normalization

    reports = []
    all_scores = []
    all_abnormal = []
    all_task_ids = []
    all_episode_indices = []
    all_action_indices = []
    all_file_indices = []
    for path in args.evaluate:
        data, metadata = load_trace(path)
        x = normalize(data, INPUT_KEYS, normalization)
        y = normalize(data, TARGET_KEYS, normalization)
        model = build_model(torch, nn, checkpoint, x.shape[1], y.shape[1])
        score = continuous_score(model, torch, x, y)
        normal = score[~metadata["abnormal"]]
        abnormal = score[metadata["abnormal"]]
        reports.append({
            "path": str(path), "rows": int(len(score)),
            "normal_rows": int(len(normal)), "abnormal_rows": int(len(abnormal)),
            "normal_score_percentiles": (
                dict(zip(("p50", "p90", "p99"), map(float, np.percentile(normal, [50, 90, 99]))))
                if len(normal) else None
            ),
            "abnormal_score_percentiles": (
                dict(zip(("p50", "p10", "p90"), map(float, np.percentile(abnormal, [50, 10, 90]))))
                if len(abnormal) else None
            ),
        })
        all_scores.append(score)
        all_abnormal.append(metadata["abnormal"])
        all_task_ids.append(metadata["task_id"])
        all_episode_indices.append(metadata["episode_idx"])
        all_action_indices.append(metadata["action_index"])
        all_file_indices.append(np.full(len(score), len(reports) - 1, dtype=np.int32))
    result = {
        "schema_version": 1,
        "track": args.track,
        "feature_policy": {
            "command": "intended_action converted by public OSC_POSE limits and 20 Hz",
            "deployment_features_only": True,
            "forbidden_features": ["task_id", "reward", "success", "disturbance_active", "object state"],
            "note": "disturbance_active is retained only as an offline evaluation label",
        },
        "adaptation_trace_count": len(args.adapt_normal),
        "reports": reports,
        "normalization": normalization if args.track == "normal_only_adaptation" else "frozen_DROID_train_only",
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(
        args.output.with_suffix(".scores.npz"),
        score=np.concatenate(all_scores), abnormal=np.concatenate(all_abnormal),
        task_id=np.concatenate(all_task_ids), episode_idx=np.concatenate(all_episode_indices),
        action_index=np.concatenate(all_action_indices), file_index=np.concatenate(all_file_indices),
    )
    print(json.dumps({"track": args.track, "files": len(reports), "rows": int(sum(map(len, all_scores)))}, indent=2))


if __name__ == "__main__":
    main()
