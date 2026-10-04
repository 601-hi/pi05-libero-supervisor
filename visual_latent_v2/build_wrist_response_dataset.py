#!/usr/bin/env python3
"""Build leakage-audited wrist response rows from traces and motion features."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def vector(row: dict, key: str, size: int) -> np.ndarray:
    value = np.asarray(row[key], dtype=np.float32)
    if value.shape != (size,):
        raise ValueError(f"{key} has shape {value.shape}, expected {(size,)}")
    return value


def condition(rows: list[dict], index: int) -> np.ndarray:
    row = rows[index]
    previous = rows[index - 1] if index else None
    previous_action = vector(previous, "intended_action", 7) if previous else np.zeros(7, dtype=np.float32)
    previous_actual = vector(previous, "actual_translation", 3) if previous else np.zeros(3, dtype=np.float32)
    quaternion = vector(row, "eef_quat_before", 4).copy()
    # q and -q denote the same orientation; canonicalization removes that discontinuity.
    if quaternion[-1] < 0:
        quaternion *= -1
    chunk_id = int(row["chunk_id"])
    same_chunk = [j for j in range(index + 1) if int(rows[j]["chunk_id"]) == chunk_id]
    chunk_phase = np.asarray([(len(same_chunk) - 1) / 4.0], dtype=np.float32)
    return np.concatenate([
        vector(row, "intended_action", 7), previous_action,
        vector(row, "eef_pos_before", 3), quaternion,
        vector(row, "joint_pos_before", 7), vector(row, "joint_vel_before", 7),
        vector(row, "gripper_qpos_before", 2), vector(row, "gripper_qvel_before", 2),
        chunk_phase, previous_actual,
    ]).astype(np.float32)


def temporal_targets(one_step: np.ndarray, horizon: int = 3) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = len(one_step)
    path = np.zeros(count, dtype=np.float32)
    net = np.zeros((count, 32), dtype=np.float32)
    valid = np.zeros(count, dtype=bool)
    for end in range(horizon - 1, count):
        window = one_step[end - horizon + 1:end + 1]
        path[end] = np.linalg.norm(window, axis=1).sum()
        net[end] = window.sum(0)
        valid[end] = True
    return path, net, valid


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", action="append", nargs=3, metavar=("NAME", "TRACE", "FEATURES"))
    parser.add_argument("--trace-root", type=Path)
    parser.add_argument("--feature-dir", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    samples = list(args.sample or [])
    if args.trace_root or args.feature_dir:
        if not (args.trace_root and args.feature_dir):
            raise ValueError("--trace-root and --feature-dir must be provided together")
        for trace in sorted(args.trace_root.glob("*.jsonl")):
            feature = args.feature_dir / f"{trace.stem}.npz"
            if not feature.exists():
                raise FileNotFoundError(feature)
            samples.append((trace.stem, str(trace), str(feature)))
    if not samples:
        raise ValueError("provide --sample or --trace-root/--feature-dir")
    output: dict[str, list[np.ndarray]] = {key: [] for key in (
        "x_condition", "one_step_grid", "one_step_valid", "step3_path", "step3_net",
        "step3_valid", "abnormal", "action_index", "sample_id",
    )}
    source_manifest = []
    for name, trace_path, feature_path in samples:
        trace = Path(trace_path); feature = Path(feature_path)
        rows = [json.loads(line) for line in trace.open(encoding="utf-8") if line.strip()]
        rows = [row for row in rows if row.get("event") == "step"]
        rows.sort(key=lambda row: int(row["action_index"]))
        data = np.load(feature, allow_pickle=False)
        actions = np.asarray([int(row["action_index"]) for row in rows])
        if not np.array_equal(actions, data["action_indices"]):
            raise ValueError(f"action mismatch for {name}")
        one_step = data["features"][:, 0, :32].astype(np.float32)
        path3, net3, valid3 = temporal_targets(one_step)
        output["x_condition"].append(np.stack([condition(rows, i) for i in range(len(rows))]))
        output["one_step_grid"].append(one_step)
        output["one_step_valid"].append(data["valid"][:, 0])
        output["step3_path"].append(path3)
        output["step3_net"].append(net3)
        output["step3_valid"].append(valid3 & data["valid"][:, 0])
        output["abnormal"].append(data["disturbance_active"])
        output["action_index"].append(actions)
        output["sample_id"].append(np.asarray([name] * len(rows)))
        source_manifest.append({"name": name, "trace": str(trace), "features": str(feature), "rows": len(rows)})
    arrays = {key: np.concatenate(parts) for key, parts in output.items()}
    metadata = {
        "condition_dim": int(arrays["x_condition"].shape[1]),
        "condition_fields": ["intended_action_7", "previous_intended_action_7", "eef_pos_before_3",
                             "canonical_eef_quat_before_4", "joint_pos_before_7", "joint_vel_before_7",
                             "gripper_qpos_before_2", "gripper_qvel_before_2", "chunk_phase_1",
                             "previous_actual_translation_3"],
        "forbidden_from_condition": ["task_id", "translation_action_scale", "disturbance_active",
                                     "reward", "done", "all_after_state", "simulator_object_ground_truth"],
        "alignment": "action_t predicts wrist_frame_t to wrist_frame_t_plus_1",
        "step3_definition": "sum of three causally aligned one-step flows, not endpoint optical flow",
        "sources": source_manifest,
    }
    arrays["metadata_json"] = np.asarray(json.dumps(metadata, ensure_ascii=False))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **arrays)
    print(json.dumps({"rows": len(arrays["x_condition"]), "condition_dim": metadata["condition_dim"],
                      "samples": len(source_manifest), "abnormal_rows": int(arrays["abnormal"].sum())},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
