#!/usr/bin/env python3
"""Run the frozen SAM2 causal identity protocol on one holdout wave."""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

# Support both ``python -m ...`` and direct script execution from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vla_supervisor.causal_object_identity import CausalMotionIdentitySelector


MODEL = "facebook/sam2.1-hiera-small"
MINIMUM_MOTION = 0.03
MINIMUM_MARGIN = 0.02
CONFIRMATION_FRAMES = 3
MAX_CANDIDATES = 12


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--public-manifest", type=Path, required=True)
    parser.add_argument("--private-map", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--wave", type=int, required=True)
    parser.add_argument("--family", default="rigid_object_transport")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def selected_jobs(public: dict, private: dict, wave: int, family: str) -> list[dict]:
    private_by_id = {row["anonymous_id"]: row for row in private["records"]}
    if set(private_by_id) != {row["anonymous_id"] for row in public["records"]}:
        raise ValueError("public and private manifest ids do not match")
    jobs = []
    for row in public["records"]:
        if row["family"] == family and row["evaluation_wave"] == wave:
            jobs.append({**row, "sidecar": private_by_id[row["anonymous_id"]]["original_sidecar"]})
    return jobs


def generate_boxes(generator, image: np.ndarray) -> tuple[list[list[float]], int]:
    from cross_suite_generalization.pilot_sam2_multicandidate_motion import select_boxes

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        generated = generator.generate(image)
    candidates = []
    pixels = image.shape[0] * image.shape[1]
    for item in generated:
        area_fraction = float(item["area"] / pixels)
        if 0.002 <= area_fraction <= 0.08:
            candidates.append({
                "bbox_xywh": [float(value) for value in item["bbox"]],
                "predicted_iou": float(item["predicted_iou"]),
                "stability_score": float(item["stability_score"]),
            })
    candidates.sort(key=lambda item: (-item["predicted_iou"], -item["stability_score"]))
    return select_boxes(candidates, MAX_CANDIDATES), len(generated)


def track(predictor, frames: np.ndarray, boxes: list[list[float]]) -> dict[int, list]:
    from cross_suite_generalization.pilot_sam2_multicandidate_motion import centroid

    tracks = {object_id: [] for object_id in range(1, len(boxes) + 1)}
    with tempfile.TemporaryDirectory(prefix="sam2_holdout_") as temporary:
        frame_dir = Path(temporary)
        for frame_index, frame in enumerate(frames):
            Image.fromarray(frame).save(frame_dir / f"{frame_index:05d}.jpg", quality=95)
        state = predictor.init_state(video_path=str(frame_dir), async_loading_frames=False)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for object_id, (x, y, width, height) in enumerate(boxes, start=1):
                predictor.add_new_points_or_box(
                    inference_state=state,
                    frame_idx=0,
                    obj_id=object_id,
                    box=np.asarray([x, y, x + width, y + height], dtype=np.float32),
                )
            for frame_index, object_ids, mask_logits in predictor.propagate_in_video(state):
                frame_masks = {
                    int(object_id): mask_logits[position, 0].detach().cpu().numpy() > 0
                    for position, object_id in enumerate(object_ids)
                }
                for object_id in tracks:
                    mask = frame_masks.get(object_id)
                    point = centroid(mask) if mask is not None else np.asarray([np.nan, np.nan])
                    tracks[object_id].append(
                        [float(point[0]), float(point[1])] if np.all(np.isfinite(point)) else None
                    )
        predictor.reset_state(state)
    return tracks


def causal_decision(tracks: dict[int, list], image_shape: tuple[int, int]) -> tuple[int | None, int | None]:
    selector = CausalMotionIdentitySelector(
        image_shape,
        minimum_motion=MINIMUM_MOTION,
        minimum_margin=MINIMUM_MARGIN,
        confirmation_frames=CONFIRMATION_FRAMES,
    )
    locked_id = lock_frame = None
    for frame_index in range(max(map(len, tracks.values()), default=0)):
        state = selector.update({object_id: points[frame_index] for object_id, points in tracks.items()})
        if locked_id is None and state.locked_object_id is not None:
            locked_id, lock_frame = state.locked_object_id, frame_index
    return locked_id, lock_frame


def render_review(output: Path, job: dict, frames: np.ndarray, boxes: list, tracks: dict, locked_id):
    indices = sorted(set((0, len(frames) // 4, len(frames) // 2, 3 * len(frames) // 4, len(frames) - 1)))
    colors = {object_id: (255, 60, 60) if object_id == locked_id else (60, 180, 255) for object_id in tracks}
    for frame_index in indices:
        image = Image.fromarray(frames[frame_index])
        draw = ImageDraw.Draw(image)
        if frame_index == 0:
            for object_id, (x, y, width, height) in enumerate(boxes, start=1):
                draw.rectangle((x, y, x + width, y + height), outline=colors[object_id], width=2)
                draw.text((x + 2, y + 2), str(object_id), fill=colors[object_id])
        for object_id, points in tracks.items():
            point = points[frame_index]
            if point is not None:
                x, y = point
                radius = 5 if object_id == locked_id else 2
                draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=colors[object_id])
        image.save(output / f"{job['anonymous_id']}_frame{frame_index:04d}.png")


def main() -> None:
    args = arguments()
    public = json.loads(args.public_manifest.read_text(encoding="utf-8"))
    private = json.loads(args.private_map.read_text(encoding="utf-8"))
    jobs = selected_jobs(public, private, args.wave, args.family)
    print(json.dumps({"wave": args.wave, "family": args.family, "jobs": len(jobs), "dry_run": args.dry_run}))
    if args.dry_run:
        for job in jobs:
            if not Path(job["sidecar"]).is_file():
                raise FileNotFoundError(job["sidecar"])
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the frozen SAM2 holdout runner")
    # Keep the manifest-selection and audit helpers importable on CPU-only
    # instances.  SAM2 is an execution-time dependency of the GPU path only.
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from sam2.sam2_video_predictor import SAM2VideoPredictor

    args.output_dir.mkdir(parents=True, exist_ok=True)
    generator = SAM2AutomaticMaskGenerator.from_pretrained(
        MODEL, points_per_side=24, pred_iou_thresh=0.72,
        stability_score_thresh=0.82, min_mask_region_area=20,
    )
    predictor = SAM2VideoPredictor.from_pretrained(MODEL)
    results = []
    for job in jobs:
        with np.load(job["sidecar"], mmap_mode="r") as sidecar:
            frames = np.asarray(sidecar["agent_images"], dtype=np.uint8)
        boxes, raw_masks = generate_boxes(generator, frames[0])
        tracks = track(predictor, frames, boxes)
        locked_id, lock_frame = causal_decision(tracks, frames.shape[1:3])
        render_review(args.output_dir, job, frames, boxes, tracks, locked_id)
        results.append({
            "anonymous_id": job["anonymous_id"],
            "locked_candidate_id": locked_id,
            "lock_frame": lock_frame,
            "num_frames": len(frames),
            "num_raw_masks": raw_masks,
            "num_candidates": len(boxes),
            "initial_boxes_xywh": boxes,
            "centroids_xy": tracks,
        })
        (args.output_dir / "predictions.partial.json").write_text(
            json.dumps({"schema_version": 1, "records": results}, indent=2) + "\n", encoding="utf-8"
        )
    final = {
        "schema_version": 1,
        "protocol": {
            "model": MODEL, "minimum_motion": MINIMUM_MOTION,
            "minimum_margin": MINIMUM_MARGIN, "confirmation_frames": CONFIRMATION_FRAMES,
            "max_candidates": MAX_CANDIDATES,
        },
        "records": results,
    }
    (args.output_dir / "predictions.json").write_text(
        json.dumps(final, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
