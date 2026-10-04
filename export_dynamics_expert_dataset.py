#!/usr/bin/env python3
"""Export leakage-controlled dynamics-expert arrays from monitoring JSONL traces."""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np

from dynamics_feature_schema import ACTION_DIM, build_condition_feature, schema_manifest


def load_rows(path: Path):
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", action="append", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--require-random-or-event", action="store_true")
    args = parser.parse_args()

    x, y, abnormal, task_ids, episode_ids, source_ids, action_indices = [], [], [], [], [], [], []
    scales, modes, successes = [], [], []
    source_manifest = []
    for source_id, path in enumerate(args.trace):
        rows = load_rows(path)
        episode_start = {}
        episode_end = {}
        groups = collections.defaultdict(list)
        for row in rows:
            key = (int(row.get("task_id", -1)), int(row.get("episode_idx", -1)))
            if row.get("event") == "episode_start": episode_start[key] = row
            elif row.get("event") == "episode_end": episode_end[key] = row
            elif row.get("event") == "step": groups[key].append(row)
        source_manifest.append({"source_id": source_id, "path": str(path), "episodes": len(groups)})
        for key, steps in sorted(groups.items()):
            steps.sort(key=lambda row: row["action_index"])
            start = episode_start.get(key, {})
            end = episode_end.get(key, {})
            mode = start.get("disturbance_start_mode", end.get("disturbance_start_mode", "legacy_unknown"))
            if args.require_random_or_event and mode not in ("random", "translation_event"):
                raise ValueError(f"{path}: episode {key} has forbidden onset mode {mode!r}")
            scale = float(start.get("translation_action_scale", 1.0))
            previous_action = np.zeros(ACTION_DIM, dtype=np.float64)
            previous_actual = np.zeros(3, dtype=np.float64)
            scenario = f"source{source_id}:task{key[0]}:episode{key[1]}"
            for row in steps:
                feature = build_condition_feature(row, previous_action, previous_actual, task_id=key[0])
                actual = np.asarray(row["actual_translation"], dtype=np.float64)
                if actual.shape != (3,) or not np.all(np.isfinite(actual)):
                    raise ValueError(f"{path}: invalid actual_translation in episode {key}")
                x.append(feature); y.append(actual); abnormal.append(bool(row.get("disturbance_active", False)))
                task_ids.append(key[0]); episode_ids.append(scenario); source_ids.append(source_id)
                action_indices.append(int(row["action_index"])); scales.append(scale); modes.append(mode)
                successes.append(bool(end.get("success", False)))
                previous_action = np.asarray(row["intended_action"], dtype=np.float64)
                previous_actual = actual

    metadata = {
        "feature_schema": schema_manifest(),
        "sources": source_manifest,
        "rows": len(x),
        "abnormal_rows": int(np.sum(abnormal)),
        "model_input_policy": "x contains deployable pre-step signals only; scale/mode/labels are metadata",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        x=np.asarray(x, dtype=np.float32),
        y=np.asarray(y, dtype=np.float32),
        abnormal=np.asarray(abnormal, dtype=bool),
        task_id=np.asarray(task_ids, dtype=np.int16),
        episode_id=np.asarray(episode_ids),
        source_id=np.asarray(source_ids, dtype=np.int16),
        action_index=np.asarray(action_indices, dtype=np.int32),
        scale=np.asarray(scales, dtype=np.float32),
        onset_mode=np.asarray(modes),
        episode_success=np.asarray(successes, dtype=bool),
        metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False)),
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
