#!/usr/bin/env python3
"""Validate outcome-blind bidirectional SAM2 output before frozen scoring."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--public", type=Path, required=True)
    args = parser.parse_args()

    result = json.loads(args.predictions.read_text(encoding="utf-8"))
    public = json.loads(args.public.read_text(encoding="utf-8"))["records"]
    records = result["records"]
    expected_ids = [row["anonymous_id"] for row in public]
    actual_ids = [row["anonymous_id"] for row in records]
    errors: list[str] = []
    if actual_ids != expected_ids:
        errors.append("anonymous id sequence differs from preregistered public manifest")
    protocol = result.get("protocol", {})
    if protocol.get("pre_frames") != 15 or protocol.get("post_frames") != 40:
        errors.append(f"unexpected tracking windows: {protocol}")
    if protocol.get("candidate_generation_uses_close_frame_only") is not True:
        errors.append("candidate generation is not declared close-frame-only")

    with_close = 0
    candidates = 0
    for record in records:
        if record.get("close_frame") is None:
            if record.get("num_candidates") != 0:
                errors.append(f"{record['anonymous_id']}: no close frame but has candidates")
            continue
        with_close += 1
        boxes = record.get("boxes_xywh", [])
        candidates += len(boxes)
        expected = {str(i) for i in range(1, len(boxes) + 1)}
        for field in (
            "pre_centroids_xy", "pre_area_fraction",
            "post_centroids_xy", "post_area_fraction",
        ):
            if set(record.get(field, {})) != expected:
                errors.append(f"{record['anonymous_id']}: {field} ids do not match boxes")
        pre_frames = record.get("pre_relative_frames", [])
        post_frames = record.get("post_relative_frames", [])
        if not pre_frames or pre_frames[-1] != 0 or len(pre_frames) < 2:
            errors.append(f"{record['anonymous_id']}: invalid pre-close frame sequence")
        if not post_frames or post_frames[0] != 0:
            errors.append(f"{record['anonymous_id']}: invalid post-close frame sequence")
        for candidate_id in expected:
            pre_len = len(record["pre_centroids_xy"][candidate_id])
            post_len = len(record["post_centroids_xy"][candidate_id])
            if pre_len != len(pre_frames) or post_len != len(post_frames):
                errors.append(
                    f"{record['anonymous_id']}: candidate {candidate_id} track/frame length mismatch"
                )
            if post_len < 30:
                errors.append(
                    f"{record['anonymous_id']}: candidate {candidate_id} cannot support "
                    "settle=5, window=25"
                )

    summary = {
        "episodes": len(records),
        "episodes_with_close": with_close,
        "candidates": candidates,
        "errors": errors,
        "valid": not errors,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
