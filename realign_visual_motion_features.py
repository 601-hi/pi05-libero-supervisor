#!/usr/bin/env python3
"""Recompute visual motion in an existing NPZ with correct action-to-frame alignment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from export_visual_motion_features import frame_feature, read_video


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    data = np.load(args.input, allow_pickle=False)
    metadata = json.loads(str(data["metadata_json"]))
    videos = metadata["videos"]
    episode_ids = data["episode_id"]
    action_indices = data["action_index"]
    visual = np.zeros_like(data["visual"], dtype=np.float32)
    valid = np.zeros(len(episode_ids), dtype=bool)

    for episode_id in np.unique(episode_ids):
        episode_id = str(episode_id)
        row_indices = np.flatnonzero(episode_ids == episode_id)
        row_indices = row_indices[np.argsort(action_indices[row_indices])]
        frames = read_video(videos[episode_id])
        if len(frames) != len(row_indices):
            raise ValueError(
                f"frame/row mismatch for {episode_id}: {len(frames)} vs {len(row_indices)}"
            )
        for position, row_index in enumerate(row_indices[:-1]):
            visual[row_index] = frame_feature(frames[position], frames[position + 1])
            valid[row_index] = True

    expected_invalid = len(np.unique(episode_ids))
    if int((~valid).sum()) != expected_invalid:
        raise AssertionError(
            f"expected {expected_invalid} invalid final actions, got {int((~valid).sum())}"
        )

    output = {name: data[name] for name in data.files if name not in {"visual", "valid", "metadata_json"}}
    metadata.update({
        "causal": True,
        "alignment": "action_j -> pre_action_frame_j to pre_action_frame_j_plus_1",
        "realigned_from": str(args.input),
    })
    output.update(
        visual=visual,
        valid=valid,
        metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False)),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **output)
    print(json.dumps({
        "rows": len(visual),
        "valid_rows": int(valid.sum()),
        "episodes": expected_invalid,
        "alignment": metadata["alignment"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
