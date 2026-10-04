#!/usr/bin/env python3
"""Track a DINO-initialized articulated counter-state with SAM2."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cross_suite_generalization.run_wrist_close_candidates_gpu import track


MODEL = "facebook/sam2.1-hiera-small"


def mask_box(mask: np.ndarray) -> list[float] | None:
    points = np.argwhere(mask)
    if not len(points):
        return None
    y0, x0 = points.min(axis=0); y1, x1 = points.max(axis=0)
    return [float(x0), float(y0), float(x1 + 1), float(y1 + 1)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    from sam2.sam2_video_predictor import SAM2VideoPredictor

    jobs = json.loads(args.jobs.read_text(encoding="utf-8"))["jobs"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictor = SAM2VideoPredictor.from_pretrained(MODEL)
    records = []
    for index, job in enumerate(jobs, 1):
        with np.load(job["sidecar"], mmap_mode="r") as data:
            frames = np.asarray(data["agent_images"], dtype=np.uint8)
        x0, y0, x1, y1 = job["anchor_box_xyxy"]
        selected = [{"box": [x0, y0, x1 - x0, y1 - y0]}]
        tracks, areas, masks = track(predictor, frames, selected, return_masks=True)
        mask_values = masks[1]
        boxes = [mask_box(mask) for mask in mask_values]
        np.savez_compressed(args.output_dir / f'{job["identity"]}_masks.npz', masks=mask_values)
        record = {
            "identity": job["identity"],
            "sidecar": job["sidecar"],
            "anchor_box_xyxy": job["anchor_box_xyxy"],
            "frames": len(frames),
            "centroids_xy": tracks[1],
            "area_fraction": areas[1],
            "boxes_xyxy": boxes,
        }
        records.append(record)
        print(json.dumps({"job": index, "identity": job["identity"], "frames": len(frames)}), flush=True)
    result = {
        "schema_version": 1,
        "model": MODEL,
        "feature_firewall": ["fixed-camera RGB", "DINO first-frame counter-state box"],
        "forbidden_inputs": ["reward", "outcome", "simulator state"],
        "records": records,
    }
    (args.output_dir / "predictions.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
