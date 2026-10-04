"""Generate task-blind SAM2 candidates and audit proximity to projected EEF."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

from vla_supervisor.gripper_projection import GripperPixelProjector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--projection-fit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frames", type=int, nargs="+", default=[40, 59, 60, 80, 120])
    parser.add_argument("--model", default="facebook/sam2.1-hiera-small")
    return parser.parse_args()


def point_to_mask_distance(mask: np.ndarray, point_xy: tuple[float, float]) -> float:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return float("inf")
    dx = xs.astype(float) - point_xy[0]
    dy = ys.astype(float) - point_xy[1]
    return float(np.sqrt(np.min(dx * dx + dy * dy)))


def bbox_iou(a: list[float], b: list[float]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x0, y0 = max(ax, bx), max(ay, by)
    x1, y1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    union = aw * ah + bw * bh - intersection
    return intersection / union if union > 0 else 0.0


def compact_candidates(generated: list[dict], shape: tuple[int, int]) -> list[dict]:
    height, width = shape
    filtered: list[dict] = []
    for item in generated:
        area_fraction = float(item["area"] / (height * width))
        x, y, box_width, box_height = (float(value) for value in item["bbox"])
        center_y = y + 0.5 * box_height
        touches_border = x <= 1 or y <= 1 or x + box_width >= width - 1 or y + box_height >= height - 1
        if not 0.002 <= area_fraction <= 0.08 or center_y < 72 or touches_border:
            continue
        candidate = {
            "bbox_xywh": [x, y, box_width, box_height],
            "area_fraction": area_fraction,
            "predicted_iou": float(item["predicted_iou"]),
            "stability_score": float(item["stability_score"]),
            "mask": np.asarray(item["segmentation"], dtype=bool),
        }
        if any(bbox_iou(candidate["bbox_xywh"], old["bbox_xywh"]) > 0.75 for old in filtered):
            continue
        filtered.append(candidate)
        if len(filtered) >= 12:
            break
    return filtered


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in args.trace.open(encoding="utf-8") if line.strip()]
    steps = [row for row in rows if row.get("event") == "step"]
    projector = GripperPixelProjector.from_fit_file(args.projection_fit)
    generator = SAM2AutomaticMaskGenerator.from_pretrained(
        args.model,
        points_per_side=24,
        pred_iou_thresh=0.72,
        stability_score_thresh=0.82,
        min_mask_region_area=20,
    )
    summary = []
    mask_payload: dict[str, np.ndarray] = {}
    with np.load(args.sidecar, mmap_mode="r") as sidecar:
        for frame_index in args.frames:
            image = np.asarray(sidecar["agent_images"][frame_index], dtype=np.uint8)
            projection = projector.project(steps[frame_index]["eef_pos_before"])
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                generated = generator.generate(image)
            candidates = compact_candidates(generated, image.shape[:2])
            for candidate_index, candidate in enumerate(candidates):
                candidate["distance_to_projected_eef_px"] = point_to_mask_distance(
                    candidate["mask"], projection.center_xy
                )
                candidate["within_projection_p90"] = (
                    candidate["distance_to_projected_eef_px"] <= projection.uncertainty_p90_px
                )
                mask_payload[f"frame{frame_index:03d}_candidate{candidate_index:02d}"] = candidate["mask"]
            candidates.sort(key=lambda item: item["distance_to_projected_eef_px"])

            canvas = image.astype(np.float32).copy()
            colors = np.random.default_rng(20260918 + frame_index).integers(48, 256, size=(len(candidates), 3))
            for rank, (candidate, color) in enumerate(zip(candidates, colors)):
                mask = candidate.pop("mask")
                canvas[mask] = 0.65 * canvas[mask] + 0.35 * color
            rendered = Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8))
            draw = ImageDraw.Draw(rendered)
            gx, gy = projection.center_xy
            radius = projection.uncertainty_p90_px
            draw.ellipse((gx - radius, gy - radius, gx + radius, gy + radius), outline=(255, 255, 255), width=2)
            draw.line((gx - 5, gy, gx + 5, gy), fill=(255, 0, 0), width=2)
            draw.line((gx, gy - 5, gx, gy + 5), fill=(255, 0, 0), width=2)
            for rank, (candidate, color) in enumerate(zip(candidates, colors)):
                x, y, width, height = candidate["bbox_xywh"]
                draw.rectangle((x, y, x + width, y + height), outline=tuple(int(v) for v in color), width=2)
                draw.text((x + 2, max(2, y + 2)), f"{rank}:{candidate['distance_to_projected_eef_px']:.1f}", fill=tuple(int(v) for v in color))
            rendered.save(args.output_dir / f"frame{frame_index:03d}_candidates.png")
            summary.append({
                "frame_index": frame_index,
                "projected_eef_xy": list(projection.center_xy),
                "projection_uncertainty_p90_px": projection.uncertainty_p90_px,
                "num_raw_masks": len(generated),
                "num_compact_candidates": len(candidates),
                "num_within_projection_p90": sum(c["within_projection_p90"] for c in candidates),
                "candidates_by_distance": candidates,
            })
    np.savez_compressed(args.output_dir / "candidate_masks.npz", **mask_payload)
    result = {
        "schema_version": 1,
        "model": args.model,
        "feature_firewall": ["fixed-view RGB", "robot EEF position", "frozen projection calibration"],
        "forbidden_inputs": ["task language", "success", "reward", "target identity"],
        "frames": summary,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps({
        "frames": len(summary),
        "raw_masks": sum(row["num_raw_masks"] for row in summary),
        "compact_candidates": sum(row["num_compact_candidates"] for row in summary),
        "within_projection_p90": sum(row["num_within_projection_p90"] for row in summary),
    }))


if __name__ == "__main__":
    main()
