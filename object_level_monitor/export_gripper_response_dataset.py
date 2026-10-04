"""Export leakage-controlled gripper-response data from monitoring traces."""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np

from dynamics_feature_schema import ACTION_DIM, build_condition_feature, remove_task_identity, schema_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", action="append", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    x, y, abnormal, episodes, action_indices, source_ids = [], [], [], [], [], []
    sources = []
    for source_id, path in enumerate(args.trace):
        groups: dict[tuple[int, int], list[dict]] = collections.defaultdict(list)
        with path.open("r", encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                if row.get("event") == "step":
                    groups[(int(row["task_id"]), int(row["episode_idx"]))].append(row)
        sources.append({"source_id": source_id, "path": str(path), "episodes": len(groups)})
        for (task_id, episode_idx), steps in sorted(groups.items()):
            steps.sort(key=lambda row: int(row["action_index"]))
            previous_action = np.zeros(ACTION_DIM, dtype=np.float64)
            previous_actual_translation = np.zeros(3, dtype=np.float64)
            episode = f"source{source_id}:task{task_id}:episode{episode_idx}"
            for row in steps:
                before = np.asarray(row["gripper_qpos_before"], dtype=np.float64)
                after = np.asarray(row["gripper_qpos_after"], dtype=np.float64)
                if before.shape != (2,) or after.shape != (2,):
                    raise ValueError(f"{path}: gripper qpos must have shape (2,)")
                feature = build_condition_feature(
                    row, previous_action, previous_actual_translation, task_id=task_id
                )
                x.append(remove_task_identity(feature))
                y.append(after - before)
                abnormal.append(bool(row.get("gripper_disturbance_applied", False)))
                episodes.append(episode)
                action_indices.append(int(row["action_index"]))
                source_ids.append(source_id)
                previous_action = np.asarray(row["intended_action"], dtype=np.float64)
                previous_actual_translation = np.asarray(row["actual_translation"], dtype=np.float64)

    metadata = {
        "rows": len(x),
        "abnormal_rows": int(np.sum(abnormal)),
        "sources": sources,
        "condition_schema": schema_manifest(),
        "condition_task_identity_used": False,
        "target": "gripper_qpos_after - gripper_qpos_before (2 joints)",
        "leakage_policy": "condition uses intended action and pre-step state only",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        x=np.asarray(x, dtype=np.float32),
        y=np.asarray(y, dtype=np.float32),
        abnormal=np.asarray(abnormal, dtype=bool),
        episode_id=np.asarray(episodes),
        action_index=np.asarray(action_indices, dtype=np.int32),
        source_id=np.asarray(source_ids, dtype=np.int16),
        metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False)),
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
