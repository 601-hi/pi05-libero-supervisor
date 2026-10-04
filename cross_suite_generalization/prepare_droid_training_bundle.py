#!/usr/bin/env python3
"""Validate exported DROID arrays and freeze train-only robust normalization."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


FORBIDDEN_FEATURE_TOKENS = ("image", "language", "reward", "success", "task", "file_path")


def robust_stats(value: np.ndarray) -> dict:
    flattened = value.reshape(len(value), -1).astype(np.float64)
    median = np.median(flattened, axis=0)
    q25 = np.percentile(flattened, 25, axis=0)
    q75 = np.percentile(flattened, 75, axis=0)
    scale = np.maximum(q75 - q25, 1e-6)
    return {
        "original_shape_without_batch": list(value.shape[1:]),
        "median": median.tolist(),
        "iqr_scale": scale.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("export_directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source_manifest = json.loads((args.export_directory / "manifest.json").read_text(encoding="utf-8"))
    train_path = args.export_directory / source_manifest["output_files"]["train"]["path"]
    calibration_path = args.export_directory / source_manifest["output_files"]["calibration"]["path"]
    train = np.load(train_path, allow_pickle=False)
    calibration = np.load(calibration_path, allow_pickle=False)
    keys = sorted(train.files)
    violations = []
    if keys != sorted(calibration.files):
        violations.append("train/calibration feature keys differ")
    if any(any(token in key.lower() for token in FORBIDDEN_FEATURE_TOKENS) for key in keys):
        violations.append("forbidden feature token present")
    for split_name, arrays in (("train", train), ("calibration", calibration)):
        lengths = {len(arrays[key]) for key in arrays.files}
        if len(lengths) != 1 or not lengths or next(iter(lengths)) == 0:
            violations.append(f"{split_name} arrays are empty or length-mismatched")
        for key in arrays.files:
            if not np.isfinite(arrays[key]).all():
                violations.append(f"{split_name}.{key} contains non-finite values")
    episode_hashes = [item["episode_identity_sha256"] for item in source_manifest["episodes"]]
    if len(episode_hashes) != len(set(episode_hashes)):
        violations.append("episode identities are not unique")
    accepted = [item for item in source_manifest["episodes"] if item["accepted"]]
    accepted_fraction = len(accepted) / max(1, len(source_manifest["episodes"]))
    if source_manifest["loaded_episodes"] != 100:
        violations.append("complete DROID-100 episode count not loaded")
    if accepted_fraction < 0.75:
        violations.append("fewer than 75% episodes pass the preregistered identification gate")
    normalization = {key: robust_stats(train[key]) for key in keys}
    decision = {
        "schema_version": 1,
        "source_manifest_sha256": hashlib.sha256(
            (args.export_directory / "manifest.json").read_bytes()
        ).hexdigest(),
        "train_npz_sha256": hashlib.sha256(train_path.read_bytes()).hexdigest(),
        "calibration_npz_sha256": hashlib.sha256(calibration_path.read_bytes()).hexdigest(),
        "loaded_episodes": source_manifest["loaded_episodes"],
        "accepted_episodes": len(accepted),
        "accepted_fraction": accepted_fraction,
        "feature_keys": keys,
        "normalization_fit_scope": "DROID train split only",
        "normalization": normalization,
        "violations": violations,
        "authorized": not violations,
        "authorized_use": (
            "domain-conditioned response-representation pretraining with DROID-domain normalization"
            if not violations else None
        ),
        "forbidden_uses": [
            "fit or tune any threshold on sealed LIBERO-10",
            "pool raw DROID controller values with raw LIBERO normalized actions",
            "treat DROID as SI-valued without further unit evidence",
            "use episode path, images, task language, reward, or success as execution-monitor features",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: decision[key] for key in (
        "loaded_episodes", "accepted_episodes", "accepted_fraction", "violations", "authorized"
    )}, indent=2))


if __name__ == "__main__":
    main()
