#!/usr/bin/env python3
"""Track close-frame wrist candidates both before and after gripper closure.

Candidate generation remains anchored at the first close frame.  Backward and
forward tracking provide a causal change-point signal: an attached object may
move relative to the wrist before contact and stabilise after contact, whereas
camera-fixed gripper parts are stable on both sides of the event.
"""
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
    MAX_AREA_FRACTION,
    MAX_CANDIDATES,
    MIN_AREA_FRACTION,
    MODEL,
    candidates,
    track,
)

PRE_FRAMES = 15
POST_FRAMES = 40


def reverse_candidate_tracks(values: dict[int, list]) -> dict[str, list]:
    return {str(key): list(reversed(track_values)) for key, track_values in values.items()}


def stringify_tracks(values: dict[int, list]) -> dict[str, list]:
    return {str(key): track_values for key, track_values in values.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--public", type=Path, required=True)
    parser.add_argument("--private", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--camera-key", choices=("wrist_images", "agent_images"),
        default="wrist_images",
        help="Camera stream to track; default preserves the original wrist protocol.",
    )
    parser.add_argument("--limit", type=int, help="Process only the first N public records (smoke testing).")
    parser.add_argument("--resume", action="store_true", help="Resume from predictions.partial.json.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    public = json.loads(args.public.read_text(encoding="utf-8"))["records"]
    if args.limit is not None:
        public = public[: args.limit]
    private = {r["anonymous_id"]: r for r in json.loads(args.private.read_text(encoding="utf-8"))["records"]}
    features = {r["anonymous_id"]: r for r in json.loads(args.features.read_text(encoding="utf-8"))["records"]}
    identifiers = {row["anonymous_id"] for row in public}
    if identifiers != set(private) or identifiers != set(features):
        raise ValueError("public/private/features anonymous ids do not match")
    missing = [private[value]["original_sidecar"] for value in identifiers
               if not Path(private[value]["original_sidecar"]).is_file()]
    if missing:
        raise FileNotFoundError(missing[0])
    if args.dry_run:
        print(json.dumps({"episodes": len(public), "camera_key": args.camera_key,
                          "with_close": sum(bool(features[value]["close_frames"]) for value in identifiers)}))
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from sam2.sam2_video_predictor import SAM2VideoPredictor

    args.output_dir.mkdir(parents=True, exist_ok=True)
    final_path = args.output_dir / "predictions.json"
    if final_path.exists():
        raise FileExistsError(final_path)
    partial_path = args.output_dir / "predictions.partial.json"

    generator = SAM2AutomaticMaskGenerator.from_pretrained(
        MODEL,
        points_per_side=24,
        pred_iou_thresh=0.72,
        stability_score_thresh=0.82,
        min_mask_region_area=20,
    )
    predictor = SAM2VideoPredictor.from_pretrained(MODEL)
    if args.resume and partial_path.exists():
        records = json.loads(partial_path.read_text(encoding="utf-8"))["records"]
    else:
        records = []
    completed_ids = {record["anonymous_id"] for record in records}
    for episode, row in enumerate(public, 1):
        identifier = row["anonymous_id"]
        if identifier in completed_ids:
            print(json.dumps({"episode": episode, "anonymous_id": identifier, "status": "already_complete"}), flush=True)
            continue
        closes = features[identifier]["close_frames"]
        if not closes:
            records.append({"anonymous_id": identifier, "close_frame": None, "num_candidates": 0})
            continue
        close = int(closes[0])
        with np.load(private[identifier]["original_sidecar"], allow_pickle=False) as data:
            wrist = np.asarray(data[args.camera_key], dtype=np.uint8)

        pre_start = max(0, close - PRE_FRAMES)
        post_stop = min(len(wrist), close + POST_FRAMES)
        # Both arrays start at the same close-frame anchor expected by track().
        pre_reverse = wrist[pre_start : close + 1][::-1]
        post = wrist[close:post_stop]
        selected, raw_count = candidates(generator, wrist[close])
        pre_tracks_reverse, pre_areas_reverse, pre_masks_reverse = track(
            predictor, pre_reverse, selected, return_masks=True
        )
        post_tracks, post_areas, post_masks = track(
            predictor, post, selected, return_masks=True
        )

        masks = np.stack([item["mask"] for item in selected]) if selected else np.zeros((0, *wrist.shape[1:3]), np.uint8)
        np.savez_compressed(
            args.output_dir / f"{identifier}_initial_masks.npz",
            masks=masks,
            boxes_xywh=np.asarray([item["box"] for item in selected], dtype=float),
        )
        np.savez_compressed(
            args.output_dir / f"{identifier}_tracked_masks.npz",
            **{
                **{f"pre_{object_id}": values[::-1] for object_id, values in pre_masks_reverse.items()},
                **{f"post_{object_id}": values for object_id, values in post_masks.items()},
            },
        )
        Image.fromarray(wrist[close]).save(args.output_dir / f"{identifier}_close_rgb.jpg", quality=95)
        records.append(
            {
                "anonymous_id": identifier,
                "goal_language": row["goal_language"],
                "close_frame": close,
                "pre_relative_frames": list(range(pre_start - close, 1)),
                "post_relative_frames": list(range(0, post_stop - close)),
                "num_raw_masks": raw_count,
                "num_candidates": len(selected),
                "boxes_xywh": [item["box"] for item in selected],
                "pre_centroids_xy": reverse_candidate_tracks(pre_tracks_reverse),
                "pre_area_fraction": reverse_candidate_tracks(pre_areas_reverse),
                "post_centroids_xy": stringify_tracks(post_tracks),
                "post_area_fraction": stringify_tracks(post_areas),
            }
        )
        partial = {"schema_version": 1, "records": records}
        partial_path.write_text(
            json.dumps(partial, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(
            json.dumps(
                {
                    "episode": episode,
                    "anonymous_id": identifier,
                    "pre_frames": len(pre_reverse),
                    "post_frames": len(post),
                    "candidates": len(selected),
                }
            ),
            flush=True,
        )

    result = {
        "schema_version": 1,
        "protocol": {
            "model": MODEL,
            "anchor": "first_close",
            "pre_frames": PRE_FRAMES,
            "post_frames": POST_FRAMES,
            "max_candidates": MAX_CANDIDATES,
            "area_fraction": [MIN_AREA_FRACTION, MAX_AREA_FRACTION],
            "candidate_generation_uses_close_frame_only": True,
            "camera_key": args.camera_key,
        },
        "records": records,
    }
    final_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
