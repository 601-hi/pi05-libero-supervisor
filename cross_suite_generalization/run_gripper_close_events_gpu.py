#!/usr/bin/env python3
"""Run bidirectional SAM2 tracking for every preregistered close event."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cross_suite_generalization.run_wrist_close_candidates_gpu import (  # noqa: E402
    MAX_AREA_FRACTION, MAX_CANDIDATES, MIN_AREA_FRACTION, MODEL, candidates, track,
)


def stringify(values: dict[int, list], reverse: bool = False) -> dict[str, list]:
    return {str(key): list(reversed(item)) if reverse else item for key, item in values.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--camera-key", choices=("wrist_images", "agent_images"), required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    events = [(episode, event) for episode in manifest["episodes"] for event in episode["events"]]
    missing = [episode["sidecar"] for episode in manifest["episodes"] if not Path(episode["sidecar"]).is_file()]
    if missing:
        raise FileNotFoundError(missing[0])
    if args.dry_run:
        print(json.dumps({"episodes": len(manifest["episodes"]), "events": len(events), "camera_key": args.camera_key}))
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from sam2.sam2_video_predictor import SAM2VideoPredictor

    args.output_dir.mkdir(parents=True, exist_ok=True)
    final_path = args.output_dir / "predictions.json"
    partial_path = args.output_dir / "predictions.partial.json"
    if final_path.exists():
        raise FileExistsError(final_path)
    generator = SAM2AutomaticMaskGenerator.from_pretrained(
        MODEL, points_per_side=24, pred_iou_thresh=0.72,
        stability_score_thresh=0.82, min_mask_region_area=20,
    )
    predictor = SAM2VideoPredictor.from_pretrained(MODEL)
    records = (
        json.loads(partial_path.read_text(encoding="utf-8"))["records"]
        if args.resume and partial_path.exists() else []
    )
    completed = {row["event_id"] for row in records}
    cached_sidecar = None
    cached_frames = None
    for index, (episode, event) in enumerate(events, 1):
        event_id = event["event_id"]
        if event_id in completed:
            continue
        if cached_sidecar != episode["sidecar"]:
            with np.load(episode["sidecar"], allow_pickle=False) as data:
                cached_frames = np.asarray(data[args.camera_key], dtype=np.uint8)
            cached_sidecar = episode["sidecar"]
        frames = cached_frames
        close = int(event["close_frame"])
        pre = frames[int(event["window_start"]): close + 1][::-1]
        post = frames[close: int(event["window_end"]) + 1]
        selected, raw_count = candidates(generator, frames[close])
        pre_tracks, pre_areas, pre_masks = track(predictor, pre, selected, return_masks=True)
        post_tracks, post_areas, post_masks = track(predictor, post, selected, return_masks=True)
        stem = event_id
        np.savez_compressed(
            args.output_dir / f"{stem}_tracked_masks.npz",
            **{
                **{f"pre_{oid}": values[::-1] for oid, values in pre_masks.items()},
                **{f"post_{oid}": values for oid, values in post_masks.items()},
            },
        )
        Image.fromarray(frames[close]).save(args.output_dir / f"{stem}_close_rgb.jpg", quality=95)
        records.append({
            **event,
            "camera_key": args.camera_key,
            "num_raw_masks": raw_count,
            "num_candidates": len(selected),
            "boxes_xywh": [item["box"] for item in selected],
            "pre_centroids_xy": stringify(pre_tracks, reverse=True),
            "pre_area_fraction": stringify(pre_areas, reverse=True),
            "post_centroids_xy": stringify(post_tracks),
            "post_area_fraction": stringify(post_areas),
        })
        partial_path.write_text(
            json.dumps({"schema_version": 1, "records": records}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({"event": index, "event_id": event_id, "candidates": len(selected)}), flush=True)
    result = {
        "schema_version": 1,
        "protocol": {
            "model": MODEL,
            "anchor": "every_preregistered_open_to_close_transition",
            "max_candidates": MAX_CANDIDATES,
            "area_fraction": [MIN_AREA_FRACTION, MAX_AREA_FRACTION],
            "camera_key": args.camera_key,
        },
        "records": records,
    }
    final_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
