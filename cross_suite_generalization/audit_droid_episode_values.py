#!/usr/bin/env python3
"""Numerically audit one verified DROID RLDS episode without inspecting images or labels."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil

import numpy as np
import tensorflow_datasets as tfds


DT_SECONDS = 1.0 / 15.0
SOURCE_SHARD = "r2d2_faceblur-train.tfrecord-00009-of-00031"


def single_shard_view(source: Path) -> Path:
    """Create TFDS indexing metadata for one unchanged, officially one-episode shard."""
    derived = source / "derived_single_shard"
    derived.mkdir(exist_ok=True)
    shutil.copy2(source / "features.json", derived / "features.json")
    info = json.loads((source / "dataset_info.json").read_text(encoding="utf-8"))
    info["splits"] = [{
        "filepathTemplate": "{DATASET}-{SPLIT}.{FILEFORMAT}-{SHARD_X_OF_Y}",
        "name": "train",
        "numBytes": str((source / SOURCE_SHARD).stat().st_size),
        "shardLengths": ["1"],
    }]
    (derived / "dataset_info.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    link = derived / "r2d2_faceblur-train.tfrecord-00000-of-00001"
    if not link.exists():
        os.symlink(source / SOURCE_SHARD, link)
    return derived


def vector_stats(values: np.ndarray) -> dict:
    return {
        "shape": list(values.shape),
        "min": np.min(values, axis=0).tolist(),
        "median": np.median(values, axis=0).tolist(),
        "max": np.max(values, axis=0).tolist(),
        "finite": bool(np.isfinite(values).all()),
    }


def lag_pair(command: np.ndarray, response: np.ndarray, lag: int) -> tuple[np.ndarray, np.ndarray]:
    start_command = max(0, -lag)
    end_command = min(len(command), len(response) - lag)
    if end_command <= start_command:
        return command[:0], response[:0]
    return command[start_command:end_command], response[start_command + lag:end_command + lag]


def alignment(command: np.ndarray, response: np.ndarray) -> dict:
    output = {}
    for lag in range(-2, 3):
        cmd, rsp = lag_pair(command, response, lag)
        cmd_flat, rsp_flat = cmd.reshape(-1), rsp.reshape(-1)
        correlation = float(np.corrcoef(cmd_flat, rsp_flat)[0, 1])
        gain = float(np.dot(cmd_flat, rsp_flat) / (np.dot(cmd_flat, cmd_flat) + 1e-12))
        residual = float(np.sqrt(np.mean((rsp_flat - gain * cmd_flat) ** 2)))
        output[str(lag)] = {
            "pairs": int(len(cmd)),
            "correlation": correlation,
            "least_squares_gain": gain,
            "rmse_after_scalar_gain": residual,
        }
    return output


def target_tracking(target: np.ndarray, state: np.ndarray) -> dict:
    output = {}
    for lag in range(0, 5):
        count = min(len(target), len(state) - lag)
        difference = state[lag:lag + count] - target[:count]
        output[str(lag)] = {
            "pairs": int(count),
            "median_l2_error": float(np.median(np.linalg.norm(difference, axis=1))),
            "p90_l2_error": float(np.percentile(np.linalg.norm(difference, axis=1), 90)),
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    derived_directory = single_shard_view(args.dataset_directory)
    builder = tfds.builder_from_directory(str(derived_directory))
    dataset = builder.as_dataset(split="train", shuffle_files=False)
    episode = next(iter(tfds.as_numpy(dataset.take(1))))
    steps = list(episode["steps"])
    observations = [step["observation"] for step in steps]
    actions = [step["action_dict"] for step in steps]

    cartesian_state = np.asarray([item["cartesian_position"] for item in observations], dtype=float)
    joint_state = np.asarray([item["joint_position"] for item in observations], dtype=float)
    gripper_state = np.asarray([item["gripper_position"] for item in observations], dtype=float)
    cartesian_velocity = np.asarray([item["cartesian_velocity"] for item in actions], dtype=float)
    joint_velocity = np.asarray([item["joint_velocity"] for item in actions], dtype=float)
    cartesian_target = np.asarray([item["cartesian_position"] for item in actions], dtype=float)
    gripper_target = np.asarray([item["gripper_position"] for item in actions], dtype=float)
    flat_action = np.asarray([step["action"] for step in steps], dtype=float)

    translation_response_velocity = np.diff(cartesian_state[:, :3], axis=0) / DT_SECONDS
    unwrapped_euler = np.unwrap(cartesian_state[:, 3:], axis=0)
    rotation_response_velocity = np.diff(unwrapped_euler, axis=0) / DT_SECONDS
    joint_response_velocity = np.diff(joint_state, axis=0) / DT_SECONDS
    translation_command = cartesian_velocity[:-1, :3]
    rotation_command = cartesian_velocity[:-1, 3:]
    joint_command = joint_velocity[:-1]

    flat_cartesian_candidate = np.concatenate((cartesian_velocity[:, :6], gripper_target), axis=1)
    flat_joint_six_candidate = np.concatenate((joint_velocity[:, :6], gripper_target), axis=1)
    flat_comparison = {
        "max_abs_difference_vs_cartesian_velocity_plus_gripper_position": float(
            np.max(np.abs(flat_action - flat_cartesian_candidate))
        ),
        "max_abs_difference_vs_first_six_joint_velocity_plus_gripper_position": float(
            np.max(np.abs(flat_action - flat_joint_six_candidate))
        ),
    }

    target_distance_before = np.linalg.norm(cartesian_target - cartesian_state, axis=1)
    target_distance_after = np.linalg.norm(cartesian_target[:-1] - cartesian_state[1:], axis=1)
    report = {
        "dataset_name": builder.info.name,
        "dataset_version": str(builder.info.version),
        "episode_count_in_loaded_shard": 1,
        "step_count": len(steps),
        "images_inspected_or_exported": False,
        "reward_success_or_language_used": False,
        "loading_view": "derived_single_shard_index_over_unchanged_official_shard_00009_of_00031",
        "assumed_control_frequency_hz": 15.0,
        "timing_source": "fixed_cadence_assumption_from_DROID_paper",
        "state": {
            "cartesian_position": vector_stats(cartesian_state),
            "joint_position": vector_stats(joint_state),
            "gripper_position": vector_stats(gripper_state),
        },
        "command": {
            "cartesian_velocity": vector_stats(cartesian_velocity),
            "joint_velocity": vector_stats(joint_velocity),
            "gripper_position": vector_stats(gripper_target),
        },
        "alignment_by_lag": {
            "translation_velocity_to_position_difference": alignment(
                translation_command, translation_response_velocity
            ),
            "angular_velocity_to_unwrapped_euler_difference": alignment(
                rotation_command, rotation_response_velocity
            ),
            "joint_velocity_to_joint_state_difference": alignment(joint_command, joint_response_velocity),
        },
        "named_action_consistency": flat_comparison,
        "cartesian_target_distance": {
            "before_mixed_pose_norm_median": float(np.median(target_distance_before)),
            "after_mixed_pose_norm_median": float(np.median(target_distance_after)),
            "warning": "This norm mixes translation and Euler rotation and is diagnostic only.",
        },
        "target_tracking_by_lag": {
            "translation_target_to_translation_state": target_tracking(
                cartesian_target[:, :3], cartesian_state[:, :3]
            ),
            "gripper_target_to_gripper_state": target_tracking(gripper_target, gripper_state),
            "rotation_target_omitted": "Euler target errors require a representation-aware periodic metric.",
        },
        "decision": "not_decided_by_script",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
