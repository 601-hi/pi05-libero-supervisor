#!/usr/bin/env python3
"""Compare homography quality in aligned fixed and wrist image streams."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vla_supervisor.background_motion import estimate_background_motion


def percentile_summary(values):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if not len(finite):
        return {"count": 0}
    return {"count": int(len(finite)), **{f"p{p:02d}": float(np.percentile(finite, p)) for p in (10, 50, 90)}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--public", type=Path, required=True)
    parser.add_argument("--private", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=5)
    args = parser.parse_args()
    if args.stride < 1:
        raise ValueError("stride must be positive")
    public = json.loads(args.public.read_text(encoding="utf-8"))["records"]
    private = {r["anonymous_id"]: r for r in json.loads(args.private.read_text(encoding="utf-8"))["records"]}
    records = []
    all_fixed, all_wrist = [], []
    reliability_counts = {"both_reliable": 0, "fixed_only": 0, "wrist_only": 0, "neither_reliable": 0}
    for index, item in enumerate(public, 1):
        identifier = item["anonymous_id"]
        with np.load(private[identifier]["original_sidecar"], allow_pickle=False) as sidecar:
            fixed = np.asarray(sidecar["agent_images"], dtype=np.uint8)
            wrist = np.asarray(sidecar["wrist_images"], dtype=np.uint8)
        if fixed.shape != wrist.shape:
            raise ValueError(f"unaligned views for {identifier}: {fixed.shape} != {wrist.shape}")
        fixed_confidence, wrist_confidence = [], []
        for frame in range(args.stride, len(fixed), args.stride):
            fixed_confidence.append(estimate_background_motion(fixed[frame-args.stride], fixed[frame]).confidence)
            wrist_confidence.append(estimate_background_motion(wrist[frame-args.stride], wrist[frame]).confidence)
        all_fixed.extend(fixed_confidence); all_wrist.extend(wrist_confidence)
        episode_reliability = dict.fromkeys(reliability_counts, 0)
        for fixed_value, wrist_value in zip(fixed_confidence, wrist_confidence):
            fixed_reliable, wrist_reliable = fixed_value >= 0.5, wrist_value >= 0.5
            key = (
                "both_reliable" if fixed_reliable and wrist_reliable else
                "fixed_only" if fixed_reliable else
                "wrist_only" if wrist_reliable else
                "neither_reliable"
            )
            reliability_counts[key] += 1
            episode_reliability[key] += 1
        records.append({
            "anonymous_id": identifier,
            "frames": int(len(fixed)),
            "sampled_pairs": len(fixed_confidence),
            "fixed_confidence": percentile_summary(fixed_confidence),
            "wrist_confidence": percentile_summary(wrist_confidence),
            "reliability_counts_at_0p5": episode_reliability,
        })
        print(json.dumps({"episode": index, "anonymous_id": identifier, "pairs": len(fixed_confidence)}), flush=True)
    result = {
        "schema_version": 1,
        "stride": args.stride,
        "episodes": len(records),
        "fixed_confidence": percentile_summary(all_fixed),
        "wrist_confidence": percentile_summary(all_wrist),
        "reliability_counts_at_0p5": reliability_counts,
        "records": records,
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("episodes", "fixed_confidence", "wrist_confidence", "reliability_counts_at_0p5")}, indent=2))


if __name__ == "__main__":
    main()
