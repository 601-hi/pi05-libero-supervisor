#!/usr/bin/env python3
"""Export allowed robot state aligned with Wave 1 visual frames."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cross_suite_generalization.diagnose_identity_v2_features import load_trace_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--private", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--trace-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    private = {row["anonymous_id"]: row for row in json.loads(args.private.read_text(encoding="utf-8"))["records"]}
    features = json.loads(args.features.read_text(encoding="utf-8"))["records"]
    records = []
    for episode in features:
        anonymous_id = episode["anonymous_id"]
        private_row = private[anonymous_id]
        with np.load(private_row["original_sidecar"], mmap_mode="r") as data:
            action_indices = np.asarray(data["action_indices"], dtype=int)
        steps = load_trace_rows(args.trace_root, private_row, action_indices)
        records.append({
            "anonymous_id": anonymous_id,
            "action_indices": action_indices.tolist(),
            "eef_position_xyz": [row["eef_pos_after"] for row in steps],
            "eef_translation_norm": [row["eef_translation_norm"] for row in steps],
            "gripper_command": [row["intended_action"][6] for row in steps],
            "gripper_qpos": [row["gripper_qpos_after"] for row in steps],
            "gripper_qvel": [row["gripper_qvel_after"] for row in steps],
        })
    args.output.write_text(json.dumps({"schema_version": 1, "records": records}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"episodes": len(records), "output": str(args.output)}))


if __name__ == "__main__":
    main()
