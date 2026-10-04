#!/usr/bin/env python3
"""Track outcome-free wrist-view candidates from the first close event."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

MODEL = "facebook/sam2.1-hiera-small"
MAX_CANDIDATES = 8
WINDOW_FRAMES = 40
MIN_AREA_FRACTION = 0.005
MAX_AREA_FRACTION = 0.65


def box_iou(left, right):
    lx, ly, lw, lh = left; rx, ry, rw, rh = right
    intersection = max(0.0, min(lx + lw, rx + rw) - max(lx, rx)) * max(0.0, min(ly + lh, ry + rh) - max(ly, ry))
    return intersection / max(lw * lh + rw * rh - intersection, 1e-9)


def candidates(generator, image):
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        generated = generator.generate(image)
    pixels = image.shape[0] * image.shape[1]
    eligible = [item for item in generated if MIN_AREA_FRACTION <= item["area"] / pixels <= MAX_AREA_FRACTION]
    eligible.sort(key=lambda item: (-float(item["predicted_iou"]), -float(item["stability_score"])))
    selected = []
    for item in eligible:
        box = [float(value) for value in item["bbox"]]
        if any(box_iou(box, previous["box"]) > 0.75 for previous in selected):
            continue
        selected.append({"box": box, "mask": np.asarray(item["segmentation"], dtype=np.uint8)})
        if len(selected) >= MAX_CANDIDATES:
            break
    return selected, len(generated)


def centroid(mask):
    points = np.argwhere(mask)
    return points.mean(axis=0)[::-1] if len(points) else np.asarray([np.nan, np.nan])


def track(predictor, frames, selected, return_masks=False):
    tracks = {index: [] for index in range(1, len(selected) + 1)}
    areas = {index: [] for index in tracks}
    mask_tracks = {index: [] for index in tracks} if return_masks else None
    with tempfile.TemporaryDirectory(prefix="sam2_wrist_close_") as temporary:
        frame_dir = Path(temporary)
        for index, frame in enumerate(frames):
            Image.fromarray(frame).save(frame_dir / f"{index:05d}.jpg", quality=95)
        state = predictor.init_state(video_path=str(frame_dir), async_loading_frames=False)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for object_id, item in enumerate(selected, 1):
                x, y, width, height = item["box"]
                predictor.add_new_points_or_box(
                    inference_state=state, frame_idx=0, obj_id=object_id,
                    box=np.asarray([x, y, x + width, y + height], dtype=np.float32),
                )
            for _, object_ids, logits in predictor.propagate_in_video(state):
                masks = {int(object_id): logits[position, 0].detach().cpu().numpy() > 0 for position, object_id in enumerate(object_ids)}
                for object_id in tracks:
                    mask = masks.get(object_id, np.zeros(frames.shape[1:3], dtype=bool))
                    point = centroid(mask)
                    tracks[object_id].append([float(point[0]), float(point[1])] if np.all(np.isfinite(point)) else None)
                    areas[object_id].append(float(mask.mean()))
                    if mask_tracks is not None:
                        mask_tracks[object_id].append(mask)
        predictor.reset_state(state)
    if mask_tracks is not None:
        return tracks, areas, {
            object_id: np.asarray(values, dtype=bool)
            for object_id, values in mask_tracks.items()
        }
    return tracks, areas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--public", type=Path, required=True)
    parser.add_argument("--private", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from sam2.sam2_video_predictor import SAM2VideoPredictor
    public = json.loads(args.public.read_text(encoding="utf-8"))["records"]
    private = {r["anonymous_id"]: r for r in json.loads(args.private.read_text(encoding="utf-8"))["records"]}
    features = {r["anonymous_id"]: r for r in json.loads(args.features.read_text(encoding="utf-8"))["records"]}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    final_path = args.output_dir / "predictions.json"
    if final_path.exists():
        raise FileExistsError(final_path)
    generator = SAM2AutomaticMaskGenerator.from_pretrained(
        MODEL, points_per_side=24, pred_iou_thresh=0.72,
        stability_score_thresh=0.82, min_mask_region_area=20,
    )
    predictor = SAM2VideoPredictor.from_pretrained(MODEL)
    records = []
    for episode, row in enumerate(public, 1):
        identifier = row["anonymous_id"]
        closes = features[identifier]["close_frames"]
        if not closes:
            records.append({"anonymous_id": identifier, "close_frame": None, "num_candidates": 0})
            continue
        close = int(closes[0])
        with np.load(private[identifier]["original_sidecar"], allow_pickle=False) as data:
            wrist = np.asarray(data["wrist_images"], dtype=np.uint8)
        window = wrist[close : min(close + WINDOW_FRAMES, len(wrist))]
        selected, raw_count = candidates(generator, window[0])
        tracks, areas = track(predictor, window, selected)
        masks = np.stack([item["mask"] for item in selected]) if selected else np.zeros((0, *window.shape[1:3]), np.uint8)
        np.savez_compressed(args.output_dir / f"{identifier}_initial_masks.npz", masks=masks,
                            boxes_xywh=np.asarray([item["box"] for item in selected], dtype=float))
        Image.fromarray(window[0]).save(args.output_dir / f"{identifier}_close_rgb.jpg", quality=95)
        records.append({
            "anonymous_id": identifier, "goal_language": row["goal_language"],
            "close_frame": close, "window_frames": int(len(window)),
            "num_raw_masks": raw_count, "num_candidates": len(selected),
            "boxes_xywh": [item["box"] for item in selected],
            "centroids_xy": {str(key): value for key, value in tracks.items()},
            "area_fraction": {str(key): value for key, value in areas.items()},
        })
        (args.output_dir / "predictions.partial.json").write_text(
            json.dumps({"schema_version": 1, "records": records}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps({"episode": episode, "anonymous_id": identifier, "frames": len(window), "candidates": len(selected)}), flush=True)
    result = {
        "schema_version": 1,
        "protocol": {"model": MODEL, "anchor": "first_close", "window_frames": WINDOW_FRAMES,
                     "max_candidates": MAX_CANDIDATES, "area_fraction": [MIN_AREA_FRACTION, MAX_AREA_FRACTION]},
        "records": records,
    }
    final_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
