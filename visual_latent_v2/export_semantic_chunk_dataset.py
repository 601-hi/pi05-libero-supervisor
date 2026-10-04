"""Build a causally aligned chunk-transition dataset from pi0.5 visual latents.

The latent captured at the start of chunk k observes the consequences of chunk
k-1.  Labels and dynamics summaries therefore always refer to the *previous*
chunk.  This prevents future-information leakage.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from two_expert_fusion import TwoExpertFusion


def parse_sources(values: list[str]) -> dict[int, Path]:
    result = {}
    for value in values:
        sid, directory = value.split("=", 1)
        result[int(sid)] = Path(directory)
    return result


def index_sidecars(directory: Path) -> dict[tuple[int, int], Path]:
    result = {}
    for path in directory.glob("*.npz"):
        with np.load(path) as data:
            key = (int(data["task_id"]), int(data["episode_idx"]))
        if key in result:
            raise ValueError(f"duplicate latent sidecar for {key}: {result[key]}, {path}")
        result[key] = path
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True, help="Dynamics dataset NPZ containing metadata_json")
    parser.add_argument("--scores", type=Path, required=True, help="Row-aligned two-expert score NPZ")
    parser.add_argument("--fusion", type=Path, required=True, help="Frozen two-expert fusion JSON")
    parser.add_argument("--latent-source", action="append", required=True, help="SOURCE_ID=LATENT_DIRECTORY")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    dataset = np.load(args.dataset, allow_pickle=False)
    scores = np.load(args.scores, allow_pickle=False)
    fusion_config = json.loads(args.fusion.read_text(encoding="utf-8"))
    if not np.array_equal(dataset["episode_id"], scores["episode_id"]):
        raise ValueError("dataset/scores episode rows do not align")
    if not np.array_equal(dataset["action_index"], scores["action_index"]):
        raise ValueError("dataset/scores action rows do not align")

    metadata = json.loads(str(dataset["metadata_json"]))
    source_dirs = parse_sources(args.latent_source)
    source_meta = {int(source["source_id"]): source for source in metadata["sources"]}
    if set(source_dirs) != set(source_meta):
        raise ValueError(f"latent source IDs {sorted(source_dirs)} != dataset source IDs {sorted(source_meta)}")

    records = []
    for source_id, latent_dir in source_dirs.items():
        trace_path = Path(source_meta[source_id]["path"])
        trace_rows = [json.loads(line) for line in trace_path.open(encoding="utf-8") if line.strip()]
        sidecars = index_sidecars(latent_dir)
        episode_keys = sorted({
            (int(row["task_id"]), int(row["episode_idx"]))
            for row in trace_rows if row.get("event") == "inference"
        })
        if set(episode_keys) != set(sidecars):
            raise ValueError(f"trace/sidecar episode mismatch for source {source_id}")

        for task_id, episode_idx in episode_keys:
            episode_id = f"source{source_id}:task{task_id}:episode{episode_idx}"
            row_indices = np.flatnonzero(dataset["episode_id"] == episode_id)
            row_indices = row_indices[np.argsort(dataset["action_index"][row_indices])]
            action_to_row = {int(dataset["action_index"][i]): int(i) for i in row_indices}
            with np.load(sidecars[(task_id, episode_idx)]) as latent:
                chunk_ids = latent["chunk_ids"].astype(np.int32)
                starts = latent["action_start_indices"].astype(np.int32)
                base = latent["base_0_rgb"].astype(np.float16)
                wrist = latent["left_wrist_0_rgb"].astype(np.float16)
                if not np.all(np.diff(chunk_ids) == 1) or not np.all(np.diff(starts) > 0):
                    raise ValueError(f"non-monotonic chunks in {sidecars[(task_id, episode_idx)]}")
                # k=0 has no previous visual transition or executed chunk.
                for k in range(1, len(chunk_ids)):
                    previous_actions = list(range(int(starts[k - 1]), int(starts[k])))
                    previous_rows = [action_to_row[a] for a in previous_actions if a in action_to_row]
                    if not previous_rows:
                        raise ValueError(f"no previous-chunk rows for {episode_id}, chunk {chunk_ids[k]}")
                    previous_rows = np.asarray(previous_rows, dtype=np.int64)
                    condition = np.asarray(dataset["x"][previous_rows], dtype=np.float32)
                    point_states = []
                    for row_index in previous_rows:
                        result = TwoExpertFusion(fusion_config).update(
                            float(scores["normal_logp"][row_index]),
                            float(scores["abnormal_logp"][row_index]),
                            int(scores["task_id"][row_index]),
                        )
                        point_states.append(result.state)
                    point_states = np.asarray(point_states)
                    records.append({
                        "episode_id": episode_id,
                        "task_id": task_id,
                        "episode_idx": episode_idx,
                        "chunk_id": int(chunk_ids[k]),
                        "observation_action_index": int(starts[k]),
                        "previous_action_start": int(starts[k - 1]),
                        "previous_action_end": int(starts[k] - 1),
                        "base_current": base[k],
                        "base_delta": (base[k].astype(np.float32) - base[k - 1].astype(np.float32)).astype(np.float16),
                        "wrist_current": wrist[k],
                        "wrist_delta": (wrist[k].astype(np.float32) - wrist[k - 1].astype(np.float32)).astype(np.float16),
                        "condition_mean": condition.mean(axis=0),
                        "condition_last": condition[-1],
                        "normal_logp_mean": float(np.mean(scores["normal_logp"][previous_rows])),
                        "abnormal_logp_mean": float(np.mean(scores["abnormal_logp"][previous_rows])),
                        "previous_active_any": bool(np.any(scores["abnormal"][previous_rows])),
                        "previous_active_fraction": float(np.mean(scores["abnormal"][previous_rows])),
                        "previous_ambiguous_any": bool(np.any(point_states == "ambiguous_overlap")),
                        "previous_ambiguous_fraction": float(np.mean(point_states == "ambiguous_overlap")),
                        "previous_known_abnormal_any": bool(np.any(point_states == "known_abnormal")),
                        "previous_unknown_abnormal_any": bool(np.any(point_states == "unknown_abnormal")),
                        "scale": float(scores["scale"][previous_rows[0]]),
                    })

    if not records:
        raise ValueError("no chunk transitions exported")
    keys = records[0].keys()
    arrays = {key: np.asarray([record[key] for record in records]) for key in keys}
    arrays["metadata_json"] = np.asarray(json.dumps({
        "schema_version": 1,
        "causal_alignment": "latent at chunk k is labeled from executed chunk k-1",
        "first_chunk_excluded": True,
        "source_latent_dirs": {str(k): str(v) for k, v in source_dirs.items()},
    }, ensure_ascii=False))
    np.savez_compressed(args.out, **arrays)
    print(json.dumps({
        "chunk_transitions": len(records),
        "episodes": len(set(arrays["episode_id"])),
        "active_transitions": int(arrays["previous_active_any"].sum()),
        "base_shape": list(arrays["base_current"].shape),
        "wrist_shape": list(arrays["wrist_current"].shape),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
