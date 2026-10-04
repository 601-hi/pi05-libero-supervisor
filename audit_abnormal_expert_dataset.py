#!/usr/bin/env python3
"""Audit trace splits before any abnormal-expert training.

This reads JSONL only, uses bounded memory per episode, and is suitable for the
small no-GPU instance. It never modifies source traces.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np


FORBIDDEN_MODEL_INPUTS = (
    "executed_action",
    "disturbance_active",
    "translation_action_scale",
    "disturbance_start_mode",
    "scheduled_disturbance_start",
    "action_index_absolute",
    "reward",
    "done",
    "success",
    "sim_contact_truth",
    "object_ground_truth_pose",
    "future_observation",
)


def parse_split(value: str) -> Tuple[str, List[Path]]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("split must be NAME=PATH_OR_GLOB")
    name, pattern = value.split("=", 1)
    path = Path(pattern)
    if any(char in pattern for char in "*?["):
        files = sorted(path.parent.glob(path.name))
    else:
        files = [path]
    if not files:
        raise argparse.ArgumentTypeError(f"split {name!r} matched no files: {pattern}")
    return name, files


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_file(path: Path) -> Dict[str, object]:
    starts, ends = {}, {}
    steps = collections.defaultdict(list)
    counts = collections.Counter()
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            event = row.get("event"); counts[event] += 1
            key = (int(row.get("task_id", -1)), int(row.get("episode_idx", -1)))
            if event == "episode_start": starts[key] = row
            elif event == "episode_end": ends[key] = row
            elif event == "step": steps[key].append(row)
    episodes = []
    for key in sorted(set(starts) | set(ends) | set(steps)):
        episode_steps = sorted(steps[key], key=lambda row: row["action_index"])
        active = [row for row in episode_steps if row.get("disturbance_active", False)]
        active_indices = [int(row["action_index"]) for row in active]
        contiguous = not active_indices or active_indices == list(range(active_indices[0], active_indices[-1] + 1))
        mean_target = float(np.mean([
            np.linalg.norm(np.asarray(row["intended_target_translation"], dtype=float)) for row in active
        ])) if active else None
        start = starts.get(key, {}); end = ends.get(key, {})
        episodes.append({
            "task_id": key[0], "episode_idx": key[1], "step_count": len(episode_steps),
            "success": end.get("success"), "env_seed": start.get("seed"),
            "sampling_noise_seed": start.get("sampling_noise_seed", end.get("sampling_noise_seed")),
            "mode": start.get("disturbance_start_mode", end.get("disturbance_start_mode")),
            "scale": start.get("translation_action_scale"),
            "scheduled_start": start.get("scheduled_disturbance_start", end.get("actual_disturbance_start")),
            "active_steps": len(active), "active_indices": active_indices,
            "active_contiguous": contiguous, "mean_active_target_norm_m": mean_target,
            "low_observability": mean_target is not None and mean_target < 0.015,
        })
    return {
        "path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size,
        "event_counts": dict(counts), "episodes": episodes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", action="append", required=True, type=parse_split, help="NAME=PATH_OR_GLOB")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    split_files: Dict[str, List[Path]] = collections.defaultdict(list)
    for name, files in args.split: split_files[name].extend(files)

    report = {"forbidden_model_inputs": list(FORBIDDEN_MODEL_INPUTS), "splits": {}, "violations": [], "warnings": []}
    identity_owners = collections.defaultdict(set)
    path_owners = collections.defaultdict(set)
    for split, files in split_files.items():
        audited = [audit_file(path) for path in files]
        episodes = [episode for item in audited for episode in item["episodes"]]
        report["splits"][split] = {
            "files": audited, "episode_count": len(episodes),
            "active_episode_count": sum(bool(e["active_steps"]) for e in episodes),
            "low_observability_count": sum(bool(e["low_observability"]) for e in episodes),
            "fixed_mode_count": sum(e["mode"] == "fixed" for e in episodes),
            "incomplete_active_count": sum(0 < e["active_steps"] < 10 for e in episodes),
            "noncontiguous_active_count": sum(not e["active_contiguous"] for e in episodes),
            "start_values": sorted({e["scheduled_start"] for e in episodes if e["scheduled_start"] is not None}),
            "scale_values": sorted({e["scale"] for e in episodes if e["scale"] is not None}),
        }
        for item in audited:
            path_owners[str(Path(item["path"]).resolve())].add(split)
            for episode in item["episodes"]:
                identity = (episode["env_seed"], episode["sampling_noise_seed"], episode["task_id"], episode["episode_idx"])
                identity_owners[identity].add(split)
                if not episode["active_contiguous"]:
                    report["violations"].append({"type": "noncontiguous_active_mask", "split": split, "episode": episode})

    for path, owners in path_owners.items():
        if len(owners) > 1: report["violations"].append({"type": "file_in_multiple_splits", "path": path, "splits": sorted(owners)})
    for identity, owners in identity_owners.items():
        if len(owners) > 1:
            report["violations"].append({"type": "scenario_identity_cross_split", "identity": identity, "splits": sorted(owners)})
    for split, summary in report["splits"].items():
        if summary["fixed_mode_count"]:
            report["warnings"].append({"type": "fixed_onset_present", "split": split, "count": summary["fixed_mode_count"]})
        if summary["incomplete_active_count"]:
            report["warnings"].append({"type": "short_fault_exposure", "split": split, "count": summary["incomplete_active_count"]})
    report["passed"] = not report["violations"]
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
