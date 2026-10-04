"""Validate visual-latent NPZ sidecars against a monitoring JSONL trace."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--latent-dir", type=Path, required=True)
    args = parser.parse_args()

    inference_rows = []
    episode_ends = {}
    with args.trace.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            key = (row.get("task_id"), row.get("episode_idx"))
            if row.get("event") == "inference":
                inference_rows.append(row)
            elif row.get("event") == "episode_end":
                episode_ends[key] = row

    npz_paths = sorted(args.latent_dir.glob("*.npz"))
    seen = set()
    checked_chunks = 0
    for path in npz_paths:
        with np.load(path) as data:
            task_id = int(data["task_id"])
            episode_idx = int(data["episode_idx"])
            key = (task_id, episode_idx)
            if key in seen:
                raise AssertionError(f"Duplicate latent artifact for {key}: {path}")
            seen.add(key)
            chunk_ids = data["chunk_ids"]
            action_starts = data["action_start_indices"]
            base = data["base_0_rgb"]
            wrist = data["left_wrist_0_rgb"]
            if not (len(chunk_ids) == len(action_starts) == len(base) == len(wrist)):
                raise AssertionError(f"Length mismatch in {path}")
            if base.shape[1:3] != (4, 4) or wrist.shape[1:3] != (4, 4):
                raise AssertionError(f"Unexpected spatial shape in {path}: {base.shape}, {wrist.shape}")
            rows = [r for r in inference_rows if (r["task_id"], r["episode_idx"]) == key]
            if len(rows) != len(chunk_ids):
                raise AssertionError(f"JSONL/NPZ chunk count mismatch for {key}: {len(rows)} != {len(chunk_ids)}")
            for i, row in enumerate(rows):
                if int(chunk_ids[i]) != int(row["chunk_id"]):
                    raise AssertionError(f"chunk_id mismatch for {key} at {i}")
                digest = hashlib.sha256(base[i].tobytes() + wrist[i].tobytes()).hexdigest()
                if digest != row["visual_latent_sha256"]:
                    raise AssertionError(f"SHA256 mismatch for {key}, chunk {chunk_ids[i]}")
                checked_chunks += 1

    expected = {key for key, row in episode_ends.items() if row.get("inference_calls", 0) > 0}
    if seen != expected:
        raise AssertionError(f"Episode coverage mismatch: missing={sorted(expected-seen)}, extra={sorted(seen-expected)}")
    print(json.dumps({"episodes": len(seen), "chunks": checked_chunks, "status": "ok"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
