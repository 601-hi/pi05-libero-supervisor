#!/usr/bin/env python3
"""Pair manually observable gripper pixels with robot-reported EEF positions."""
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
    result = []
    for episode in features:
        anonymous_id = episode["anonymous_id"]
        private_row = private[anonymous_id]
        with np.load(private_row["original_sidecar"], mmap_mode="r") as data:
            action_indices = np.asarray(data["action_indices"], dtype=int)
        steps = load_trace_rows(args.trace_root, private_row, action_indices)
        close = episode["close_frames"][0]
        result.append({
            "anonymous_id": anonymous_id,
            "close_frame": close,
            "eef_position_xyz": steps[close]["eef_pos_after"],
            "gripper_center_xy": None,
            "annotation_confidence": None,
        })
    args.output.write_text(json.dumps({"schema_version": 1, "records": result}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"pairs": len(result), "output": str(args.output)}))


if __name__ == "__main__":
    main()
