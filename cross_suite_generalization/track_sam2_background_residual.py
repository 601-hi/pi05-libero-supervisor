"""Track task-blind masks through a stall window and measure background residual."""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sam2.sam2_video_predictor import SAM2VideoPredictor

from vla_supervisor.background_motion import estimate_background_motion, residual_motion_in_mask


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--masks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", type=int, default=40)
    parser.add_argument("--seed", type=int, default=59)
    parser.add_argument("--end", type=int, default=130)
    parser.add_argument("--model", default="facebook/sam2.1-hiera-small")
    return parser.parse_args()


def center(mask: np.ndarray) -> list[float] | None:
    points = np.argwhere(mask)
    if not len(points):
        return None
    yx = points.mean(axis=0)
    return [float(yx[1]), float(yx[0])]


def main() -> None:
    config = args()
    with np.load(config.sidecar, mmap_mode="r") as sidecar:
        frames = np.asarray(sidecar["agent_images"][config.start : config.end + 1], dtype=np.uint8)
    seed_relative = config.seed - config.start
    with np.load(config.masks) as archive:
        keys = sorted(key for key in archive.files if key.startswith(f"frame{config.seed:03d}_"))
        seed_masks = [np.asarray(archive[key], dtype=bool) for key in keys]
    if not seed_masks:
        raise RuntimeError("no seed masks found")

    predictor = SAM2VideoPredictor.from_pretrained(config.model)
    with tempfile.TemporaryDirectory(prefix="stall_track_") as temporary:
        frame_dir = Path(temporary)
        for index, frame in enumerate(frames):
            Image.fromarray(frame).save(frame_dir / f"{index:05d}.jpg", quality=95)
        state = predictor.init_state(video_path=str(frame_dir), async_loading_frames=False)
        tracked: dict[int, dict[int, np.ndarray]] = {}
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for object_id, mask in enumerate(seed_masks, start=1):
                predictor.add_new_mask(state, seed_relative, object_id, mask)
            for frame_index, object_ids, logits in predictor.propagate_in_video(
                state, start_frame_idx=seed_relative, max_frame_num_to_track=len(frames) - seed_relative
            ):
                tracked[int(frame_index)] = {
                    int(object_id): (logits[position, 0].detach().cpu().numpy() > 0)
                    for position, object_id in enumerate(object_ids)
                }
        predictor.reset_state(state)

    records = []
    for object_id in range(1, len(seed_masks) + 1):
        per_step = []
        previous_center = None
        for relative in range(seed_relative, len(frames) - 1):
            mask = tracked.get(relative, {}).get(object_id)
            if mask is None:
                continue
            estimate = estimate_background_motion(frames[relative], frames[relative + 1])
            residual = residual_motion_in_mask(frames[relative], frames[relative + 1], mask, estimate)
            current_center = center(mask)
            displacement = None
            if previous_center is not None and current_center is not None:
                displacement = float(np.linalg.norm(np.asarray(current_center) - np.asarray(previous_center)))
            previous_center = current_center
            per_step.append({
                "action_index": config.start + relative,
                "background_valid": estimate.valid,
                "background_confidence": estimate.confidence,
                "center_xy": current_center,
                "centroid_step_px": displacement,
                "median_residual_px": residual.median_px if residual.valid_pixels else None,
                "p90_residual_px": residual.p90_px if residual.valid_pixels else None,
                "coherent_fraction": residual.coherent_fraction if residual.valid_pixels else None,
                "valid_pixels": residual.valid_pixels,
            })
        valid_residual = [row["median_residual_px"] for row in per_step if row["median_residual_px"] is not None]
        coherent = [row["coherent_fraction"] for row in per_step if row["coherent_fraction"] is not None]
        records.append({
            "object_id": object_id,
            "seed_key": keys[object_id - 1],
            "steps": len(per_step),
            "median_of_median_residual_px": float(np.median(valid_residual)) if valid_residual else None,
            "p90_of_median_residual_px": float(np.percentile(valid_residual, 90)) if valid_residual else None,
            "median_coherent_fraction": float(np.median(coherent)) if coherent else None,
            "per_step": per_step,
        })
    payload = {
        "schema_version": 1,
        "model": config.model,
        "feature_firewall": ["fixed-view RGB", "task-blind SAM2 seed masks"],
        "window": {"start": config.start, "seed": config.seed, "end": config.end},
        "warning": "Raw residuals are measurements, not anomaly probabilities.",
        "candidates": records,
    }
    config.output.parent.mkdir(parents=True, exist_ok=True)
    config.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({
        "candidates": len(records),
        "tracked_steps": sum(row["steps"] for row in records),
        "residual_medians": [row["median_of_median_residual_px"] for row in records],
    }))


if __name__ == "__main__":
    main()
