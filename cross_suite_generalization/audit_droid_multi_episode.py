#!/usr/bin/env python3
"""Audit command/response alignment across verified DROID shards without images or labels."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil

import numpy as np
import tensorflow_datasets as tfds

from delayed_response_likelihood import episode_alignment_quality, fit_lag_bank


DT_SECONDS = 1.0 / 15.0
SHARD_GLOB = "r2d2_faceblur-train.tfrecord-*-of-00031"


def make_view(source: Path) -> tuple[Path, list[Path], list[int]]:
    shards = sorted(source.glob(SHARD_GLOB))
    if not shards:
        raise FileNotFoundError(f"no verified shards matching {SHARD_GLOB} in {source}")
    original_info = json.loads((source / "dataset_info.json").read_text(encoding="utf-8"))
    official_lengths = [int(value) for value in original_info["splits"][0]["shardLengths"]]
    selected_lengths = [official_lengths[int(path.name.split("-")[-3])] for path in shards]

    derived = source / "derived_selected_shards"
    derived.mkdir(exist_ok=True)
    shutil.copy2(source / "features.json", derived / "features.json")
    info = dict(original_info)
    info["splits"] = [{
        "filepathTemplate": "{DATASET}-{SPLIT}.{FILEFORMAT}-{SHARD_X_OF_Y}",
        "name": "train",
        "numBytes": str(sum(path.stat().st_size for path in shards)),
        "shardLengths": [str(value) for value in selected_lengths],
    }]
    (derived / "dataset_info.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    for index, shard in enumerate(shards):
        link = derived / f"r2d2_faceblur-train.tfrecord-{index:05d}-of-{len(shards):05d}"
        if link.is_symlink() and link.resolve() != shard.resolve():
            link.unlink()
        if not link.exists():
            os.symlink(shard, link)
    return derived, shards, selected_lengths


def streaming_dataset(builder):
    """Read one shard at a time and keep image payloads compressed and unused."""
    skip = tfds.decode.SkipDecoding()
    decoders = {"steps": {"observation": {
        "wrist_image_left": skip,
        "exterior_image_1_left": skip,
        "exterior_image_2_left": skip,
    }}}
    read_config = tfds.ReadConfig(
        try_autocache=False,
        num_parallel_calls_for_decode=1,
        num_parallel_calls_for_interleave_files=1,
        interleave_cycle_length=1,
    )
    return builder.as_dataset(
        split="train", shuffle_files=False, decoders=decoders, read_config=read_config
    )


def lag_pair(command: np.ndarray, response: np.ndarray, lag: int) -> tuple[np.ndarray, np.ndarray]:
    start = max(0, -lag)
    end = min(len(command), len(response) - lag)
    return command[start:end], response[start + lag:end + lag]


def sufficient_stats(command: np.ndarray, response: np.ndarray) -> dict[str, float]:
    x, y = command.reshape(-1), response.reshape(-1)
    return {
        "n": int(len(x)), "sx": float(x.sum()), "sy": float(y.sum()),
        "sxx": float(x @ x), "syy": float(y @ y), "sxy": float(x @ y),
    }


def combine(stats: list[dict[str, float]]) -> dict[str, float]:
    total = {key: sum(item[key] for item in stats) for key in stats[0]}
    n, sx, sy = total["n"], total["sx"], total["sy"]
    covariance = total["sxy"] - sx * sy / n
    variance_x = total["sxx"] - sx * sx / n
    variance_y = total["syy"] - sy * sy / n
    gain = total["sxy"] / (total["sxx"] + 1e-12)
    squared_error = total["syy"] - 2 * gain * total["sxy"] + gain * gain * total["sxx"]
    return {
        "scalar_pairs": int(n),
        "correlation": float(covariance / np.sqrt(variance_x * variance_y + 1e-12)),
        "least_squares_gain": float(gain),
        "rmse_after_scalar_gain": float(np.sqrt(max(0.0, squared_error) / n)),
    }


def episode_arrays(episode: dict) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict[str, np.ndarray]]:
    steps = list(episode["steps"])
    obs = [step["observation"] for step in steps]
    act = [step["action_dict"] for step in steps]
    cart = np.asarray([item["cartesian_position"] for item in obs], dtype=float)
    joint = np.asarray([item["joint_position"] for item in obs], dtype=float)
    cart_velocity = np.asarray([item["cartesian_velocity"] for item in act], dtype=float)
    joint_velocity = np.asarray([item["joint_velocity"] for item in act], dtype=float)
    modalities = {
        "translation": (cart_velocity[:-1, :3], np.diff(cart[:, :3], axis=0) / DT_SECONDS),
        "rotation_unwrapped_euler": (
            cart_velocity[:-1, 3:], np.diff(np.unwrap(cart[:, 3:], axis=0), axis=0) / DT_SECONDS
        ),
        "joint": (joint_velocity[:-1], np.diff(joint, axis=0) / DT_SECONDS),
    }
    targets = {
        "eef_translation": np.asarray([item["cartesian_position"] for item in act], dtype=float)[:, :3],
        "joint_position": np.asarray([item["joint_position"] for item in act], dtype=float),
        "gripper_position": np.asarray([item["gripper_position"] for item in act], dtype=float),
        "eef_translation_state": cart[:, :3],
        "joint_position_state": joint,
        "gripper_position_state": np.asarray(
            [item["gripper_position"] for item in obs], dtype=float
        ),
    }
    return modalities, targets


def target_error_by_lag(target: np.ndarray, state: np.ndarray, maximum_lag: int = 8) -> dict:
    output = {}
    for lag in range(maximum_lag + 1):
        count = min(len(target), len(state) - lag)
        error = np.linalg.norm(state[lag:lag + count] - target[:count], axis=1)
        output[str(lag)] = {
            "pairs": int(count),
            "median_l2_error": float(np.median(error)),
            "p90_l2_error": float(np.percentile(error, 90)),
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    view, shards, expected_lengths = make_view(args.dataset_directory)
    dataset = streaming_dataset(tfds.builder_from_directory(str(view)))

    lags = range(-3, 6)
    deployable_lags = range(0, 6)
    pooled = {name: {lag: [] for lag in lags} for name in ("translation", "rotation_unwrapped_euler", "joint")}
    best_lags = {name: [] for name in pooled}
    step_counts = []
    episode_diagnostics = []
    modality_episodes = {name: [] for name in pooled}
    for episode_index, episode in enumerate(tfds.as_numpy(dataset)):
        modalities, targets = episode_arrays(episode)
        step_counts.append(len(list(episode["steps"])))
        episode_report = {
            "episode_index_in_selected_view": episode_index,
            "modalities": {},
            "absolute_target_tracking": {
                "eef_translation": target_error_by_lag(
                    targets["eef_translation"], targets["eef_translation_state"]
                ),
                "joint_position": target_error_by_lag(
                    targets["joint_position"], targets["joint_position_state"]
                ),
                "gripper_position": target_error_by_lag(
                    targets["gripper_position"], targets["gripper_position_state"]
                ),
            },
        }
        for name, (command, response) in modalities.items():
            modality_episodes[name].append((command, response))
            episode_scores = {}
            for lag in lags:
                x, y = lag_pair(command, response, lag)
                stats = sufficient_stats(x, y)
                pooled[name][lag].append(stats)
                episode_scores[lag] = combine([stats])["correlation"]
            best_lags[name].append(max(episode_scores, key=episode_scores.get))
            episode_report["modalities"][name] = {
                "command_rms": float(np.sqrt(np.mean(command ** 2))),
                "response_rms": float(np.sqrt(np.mean(response ** 2))),
                "correlation_by_lag": {str(lag): value for lag, value in episode_scores.items()},
                "best_lag": int(max(episode_scores, key=episode_scores.get)),
                "best_correlation": float(max(episode_scores.values())),
            }
        episode_diagnostics.append(episode_report)

    identification_quality = {
        name: [episode_alignment_quality(command, response, lags) for command, response in episodes]
        for name, episodes in modality_episodes.items()
    }
    accepted_lag_banks = {}
    for name, episodes in modality_episodes.items():
        accepted = [
            episode for episode, quality in zip(episodes, identification_quality[name])
            if quality["accepted_for_identification"]
        ]
        accepted_lag_banks[name] = [
            component.__dict__ for component in fit_lag_bank(accepted, deployable_lags)
        ]

    report = {
        "dataset": "DROID-100/r2d2_faceblur/1.0.0",
        "source_shards": [path.name for path in shards],
        "source_shard_bytes": [path.stat().st_size for path in shards],
        "official_expected_episodes": int(sum(expected_lengths)),
        "loaded_episodes": len(step_counts),
        "step_counts": step_counts,
        "assumed_frequency_hz": 15.0,
        "images_inspected_or_exported": False,
        "reward_success_or_language_used": False,
        "alignment_by_lag": {
            name: {str(lag): combine(values) for lag, values in by_lag.items()}
            for name, by_lag in pooled.items()
        },
        "per_episode_best_lag_distribution": {
            name: {str(lag): values.count(lag) for lag in lags}
            for name, values in best_lags.items()
        },
        "episode_diagnostics": episode_diagnostics,
        "identification_quality": identification_quality,
        "lag_bank_fit_on_quality_accepted_episodes_only": accepted_lag_banks,
        "lag_bank_scope_warning": (
            "Transparent pilot fit on this small external sample only. It is not a frozen detector, "
            "and its weights must not be tuned on the final LIBERO holdout."
        ),
        "interpretation_guardrail": (
            "Positive lag compares command t with measured response t+lag to t+lag+1. "
            "This audit may falsify zero-lag alignment but does not by itself establish units, frames, or causality."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
