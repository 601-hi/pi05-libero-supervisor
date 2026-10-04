#!/usr/bin/env python3
"""Export quality-gated DROID low-dimensional dynamics arrays without oracle features."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from delayed_response_likelihood import episode_alignment_quality


HISTORY_LAGS = tuple(range(6))
DIAGNOSTIC_LAGS = tuple(range(-3, 6))
DT_SECONDS = 1.0 / 15.0


def split_for(identity: str) -> str:
    """Stable episode-level 80/20 split; identity is never exported as a feature."""
    bucket = int.from_bytes(hashlib.sha256(identity.encode("utf-8")).digest()[:8], "big") % 5
    return "calibration" if bucket == 0 else "train"


def wrap_angle_difference(after: np.ndarray, before: np.ndarray) -> np.ndarray:
    return (after - before + np.pi) % (2 * np.pi) - np.pi


def episode_identity(episode: dict, fallback_index: int) -> str:
    metadata = episode.get("episode_metadata", {})
    raw = metadata.get("file_path", f"fallback-index-{fallback_index}")
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="strict")
    return str(raw)


def main() -> None:
    import tensorflow_datasets as tfds
    from audit_droid_multi_episode import episode_arrays, make_view, streaming_dataset

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_directory", type=Path)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    view, shards, expected_lengths = make_view(args.dataset_directory)
    dataset = streaming_dataset(tfds.builder_from_directory(str(view)))
    rows = {split: {key: [] for key in (
        "command_history_cartesian_velocity", "state_eef_pose_xyz_euler",
        "state_joint_position", "state_gripper_position", "response_eef_delta_xyz_euler",
        "response_joint_delta", "response_gripper_delta",
    )} for split in ("train", "calibration")}
    episodes_report = []

    for episode_index, episode in enumerate(tfds.as_numpy(dataset)):
        steps = list(episode["steps"])
        modalities, _ = episode_arrays(episode)
        quality = {
            name: episode_alignment_quality(command, response, DIAGNOSTIC_LAGS)
            for name, (command, response) in modalities.items()
        }
        accepted = all(item["accepted_for_identification"] for item in quality.values())
        identity = episode_identity(episode, episode_index)
        split = split_for(identity)
        exported = 0
        if accepted:
            obs = [step["observation"] for step in steps]
            act = [step["action_dict"] for step in steps]
            cart_state = np.asarray([item["cartesian_position"] for item in obs], dtype=float)
            joint_state = np.asarray([item["joint_position"] for item in obs], dtype=float)
            gripper_state = np.asarray([item["gripper_position"] for item in obs], dtype=float)
            command = np.asarray([item["cartesian_velocity"] for item in act], dtype=float)
            for response_index in range(max(HISTORY_LAGS), len(steps) - 1):
                history = np.stack([command[response_index - lag] for lag in HISTORY_LAGS])
                eef_delta = np.concatenate((
                    cart_state[response_index + 1, :3] - cart_state[response_index, :3],
                    wrap_angle_difference(
                        cart_state[response_index + 1, 3:], cart_state[response_index, 3:]
                    ),
                ))
                destination = rows[split]
                destination["command_history_cartesian_velocity"].append(history)
                destination["state_eef_pose_xyz_euler"].append(cart_state[response_index])
                destination["state_joint_position"].append(joint_state[response_index])
                destination["state_gripper_position"].append(gripper_state[response_index])
                destination["response_eef_delta_xyz_euler"].append(eef_delta)
                destination["response_joint_delta"].append(
                    joint_state[response_index + 1] - joint_state[response_index]
                )
                destination["response_gripper_delta"].append(
                    gripper_state[response_index + 1] - gripper_state[response_index]
                )
                exported += 1
        episodes_report.append({
            "episode_identity_sha256": hashlib.sha256(identity.encode("utf-8")).hexdigest(),
            "split": split,
            "step_count": len(steps),
            "accepted": accepted,
            "exported_transitions": exported,
            "quality": quality,
        })

    args.output_directory.mkdir(parents=True, exist_ok=True)
    output_files = {}
    for split, arrays in rows.items():
        path = args.output_directory / f"droid100_dynamics_{split}.npz"
        packed = {name: np.asarray(values, dtype=np.float32) for name, values in arrays.items()}
        np.savez_compressed(path, **packed)
        output_files[split] = {
            "path": path.name,
            "transitions": int(len(next(iter(packed.values())))),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "shapes": {name: list(value.shape) for name, value in packed.items()},
        }
    manifest = {
        "schema_version": 1,
        "source": "DROID-100/r2d2_faceblur/1.0.0 complete 31 shards",
        "source_shard_count": len(shards),
        "official_expected_episodes": int(sum(expected_lengths)),
        "loaded_episodes": len(episodes_report),
        "episode_split": "sha256(file_path) modulo 5; bucket 0 calibration, others train",
        "history_lags": list(HISTORY_LAGS),
        "frequency_assumption_hz": 15.0,
        "images_used": False,
        "language_reward_success_used": False,
        "episode_identity_exported_as_feature": False,
        "eef_rotation_delta": "componentwise wrapped Euler difference; representation-specific baseline",
        "output_files": output_files,
        "episodes": episodes_report,
        "training_authorization": False,
        "training_blocker": (
            "Export is mechanically ready, but command units/reference frame and mixed episode control "
            "semantics remain evidence-gated. Enabling training requires a separate signed decision manifest."
        ),
    }
    manifest_path = args.output_directory / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "loaded_episodes": manifest["loaded_episodes"],
        "accepted_episodes": sum(item["accepted"] for item in episodes_report),
        "rejected_episodes": sum(not item["accepted"] for item in episodes_report),
        "output_files": output_files,
        "training_authorization": False,
    }, indent=2))


if __name__ == "__main__":
    main()
