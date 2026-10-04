#!/usr/bin/env python3
"""Reconstruct and save initial SAM2 masks with strict candidate-ID checks."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


MODEL = "facebook/sam2.1-hiera-small"
MAX_CANDIDATES = 12
COLORS = (
    (0, 255, 255), (255, 215, 0), (255, 105, 180), (50, 205, 50),
    (255, 140, 0), (138, 43, 226), (0, 191, 255), (255, 99, 71),
    (127, 255, 0), (238, 130, 238), (64, 224, 208), (255, 255, 255),
)


def box_iou(left: list[float], right: list[float]) -> float:
    lx, ly, lw, lh = left
    rx, ry, rw, rh = right
    intersection = max(0.0, min(lx + lw, rx + rw) - max(lx, rx)) * max(0.0, min(ly + lh, ry + rh) - max(ly, ry))
    return intersection / max(lw * lh + rw * rh - intersection, 1e-9)


def select_items(generated: list[dict], image_shape: tuple[int, int], max_candidates: int = MAX_CANDIDATES) -> list[dict]:
    pixels = image_shape[0] * image_shape[1]
    candidates = []
    for item in generated:
        area_fraction = float(item["area"] / pixels)
        if 0.002 <= area_fraction <= 0.08:
            candidates.append(item)
    candidates.sort(key=lambda item: (-float(item["predicted_iou"]), -float(item["stability_score"])))
    selected = []
    selected_boxes = []
    for item in candidates:
        box = [float(value) for value in item["bbox"]]
        x, y, width, height = box
        center_y = y + height / 2
        if center_y < 72 or y <= 1 or y + height >= 223:
            continue
        if any(box_iou(box, previous) > 0.75 for previous in selected_boxes):
            continue
        selected.append(item)
        selected_boxes.append(box)
        if len(selected) >= max_candidates:
            break
    return selected


def render_sheet(frame: np.ndarray, items: list[dict], output: Path) -> None:
    source = Image.fromarray(frame).convert("RGB")
    tiles = []
    font = ImageFont.load_default()
    for candidate_id, item in enumerate(items, start=1):
        color = COLORS[(candidate_id - 1) % len(COLORS)]
        mask = np.asarray(item["segmentation"], dtype=bool)
        canvas = frame.astype(float).copy()
        canvas[mask] = 0.35 * canvas[mask] + 0.65 * np.asarray(color)
        x, y, width, height = map(float, item["bbox"])
        margin = max(12.0, 0.6 * max(width, height))
        left, top = max(0, int(x - margin)), max(0, int(y - margin))
        right, bottom = min(source.width, int(x + width + margin)), min(source.height, int(y + height + margin))
        crop = Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8)).crop((left, top, right, bottom))
        scale = min(150 / max(crop.width, 1), 130 / max(crop.height, 1))
        crop = crop.resize((max(1, int(crop.width * scale)), max(1, int(crop.height * scale))))
        tile = Image.new("RGB", (160, 160), (24, 24, 24))
        tile.paste(crop, ((160 - crop.width) // 2, 24 + (130 - crop.height) // 2))
        draw = ImageDraw.Draw(tile)
        draw.text((5, 5), f"candidate {candidate_id}", fill=color, font=font)
        tiles.append(tile)
    columns = 4
    rows = (len(tiles) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * 160, rows * 160), (15, 15, 15))
    for index, tile in enumerate(tiles):
        sheet.paste(tile, ((index % columns) * 160, (index // columns) * 160))
    sheet.save(output, quality=95)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--public", type=Path, required=True)
    parser.add_argument("--private", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    public = {row["anonymous_id"]: row for row in json.loads(args.public.read_text(encoding="utf-8"))["records"]}
    private = {row["anonymous_id"]: row for row in json.loads(args.private.read_text(encoding="utf-8"))["records"]}
    predictions = json.loads(args.predictions.read_text(encoding="utf-8"))["records"]
    for row in predictions:
        if row["anonymous_id"] not in public or row["anonymous_id"] not in private:
            raise ValueError(f"manifest mismatch for {row['anonymous_id']}")
        if not Path(private[row["anonymous_id"]]["original_sidecar"]).is_file():
            raise FileNotFoundError(private[row["anonymous_id"]]["original_sidecar"])
    print(json.dumps({"episodes": len(predictions), "dry_run": args.dry_run}))
    if args.dry_run:
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to reconstruct SAM2 masks")
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

    generator = SAM2AutomaticMaskGenerator.from_pretrained(
        MODEL, points_per_side=24, pred_iou_thresh=0.72,
        stability_score_thresh=0.82, min_mask_region_area=20,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for episode_index, row in enumerate(predictions, start=1):
        identifier = row["anonymous_id"]
        with np.load(private[identifier]["original_sidecar"], mmap_mode="r") as data:
            frame = np.asarray(data["agent_images"][0], dtype=np.uint8)
        Image.fromarray(frame).save(args.output_dir / f"{identifier}_initial_rgb.jpg", quality=98)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            generated = generator.generate(frame)
        selected = select_items(generated, frame.shape[:2])
        boxes = np.asarray([item["bbox"] for item in selected], dtype=float)
        expected = np.asarray(row["initial_boxes_xywh"], dtype=float)
        if boxes.shape != expected.shape or not np.allclose(boxes, expected, atol=1e-6, rtol=0):
            raise RuntimeError(f"candidate ID mismatch for {identifier}: reconstructed boxes differ")
        masks = np.stack([np.asarray(item["segmentation"], dtype=np.uint8) for item in selected])
        np.savez_compressed(args.output_dir / f"{identifier}_initial_masks.npz", masks=masks, boxes_xywh=boxes)
        render_sheet(frame, selected, args.output_dir / f"{identifier}_mask_sheet.jpg")
        print(json.dumps({"episode": episode_index, "anonymous_id": identifier, "candidates": len(selected)}), flush=True)


if __name__ == "__main__":
    main()
