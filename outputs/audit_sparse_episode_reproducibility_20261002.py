#!/usr/bin/env python3
"""Audit whether two nominally identical LIBERO episodes share a causal prefix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load(path: Path, episode_idx: int, event: str):
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("episode_idx") == episode_idx and row.get("event") == event:
                rows.append(row)
    return rows


def vector(row, key, *, nested=False):
    if nested:
        row = row.get("recovery_checkpoint_evidence") or {}
    value = row.get(key)
    return None if value is None else np.asarray(value, dtype=float)


def max_difference(left, right, key, *, nested=False):
    values = []
    first = None
    for index, (a, b) in enumerate(zip(left, right)):
        va, vb = vector(a, key, nested=nested), vector(b, key, nested=nested)
        if va is None or vb is None or va.shape != vb.shape:
            difference = float("inf") if (va is None) != (vb is None) else 0.0
        else:
            difference = float(np.max(np.abs(va - vb)))
        values.append(difference)
        if first is None and difference > 1e-12:
            first = index
    return (max(values, default=0.0), first)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--episode", type=int, default=2)
    parser.add_argument("--prefix-steps", type=int, default=82)
    args = parser.parse_args()

    left = load(args.reference, args.episode, "step")[:args.prefix_steps]
    right = load(args.candidate, args.episode, "step")[:args.prefix_steps]
    inference_left = load(args.reference, args.episode, "inference")
    inference_right = load(args.candidate, args.episode, "inference")
    report = {
        "episode_idx": args.episode,
        "requested_prefix_steps": args.prefix_steps,
        "compared_steps": min(len(left), len(right)),
        "step_counts_equal": len(left) == len(right) == args.prefix_steps,
    }
    for key, nested in (
        ("intended_action", False),
        ("eef_pos_before", False),
        ("eef_pos_after", False),
        ("joint_pos", True),
        ("minimum_self_clearance_m", True),
        ("minimum_environment_clearance_m", True),
    ):
        maximum, first = max_difference(left, right, key, nested=nested)
        report[key] = {"maximum_absolute_difference": maximum,
                       "first_difference_index": first}
    hashes_left = [row.get("sampling_noise_sha256") for row in inference_left]
    hashes_right = [row.get("sampling_noise_sha256") for row in inference_right]
    report["noise_hash_prefix_equal"] = (
        hashes_left[:min(len(hashes_left), len(hashes_right))]
        == hashes_right[:min(len(hashes_left), len(hashes_right))])
    report["causal_prefix_exact"] = (
        report["step_counts_equal"]
        and report["noise_hash_prefix_equal"]
        and all(report[key]["maximum_absolute_difference"] == 0.0 for key in (
            "intended_action", "eef_pos_before", "eef_pos_after", "joint_pos")))
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
