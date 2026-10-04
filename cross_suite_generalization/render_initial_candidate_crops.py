#!/usr/bin/env python3
"""Render enlarged initial candidate crops for motion-independent identity labels."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--public", type=Path, required=True)
    parser.add_argument("--private", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    public = {row["anonymous_id"]: row for row in json.loads(args.public.read_text(encoding="utf-8"))["records"]}
    private = {row["anonymous_id"]: row for row in json.loads(args.private.read_text(encoding="utf-8"))["records"]}
    predictions = json.loads(args.predictions.read_text(encoding="utf-8"))["records"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default()
    for row in predictions:
        identifier = row["anonymous_id"]
        with np.load(private[identifier]["original_sidecar"], mmap_mode="r") as data:
            frame = np.asarray(data["agent_images"][0], dtype=np.uint8)
        source = Image.fromarray(frame).convert("RGB")
        tiles = []
        for candidate_id, (x, y, width, height) in enumerate(row["initial_boxes_xywh"], start=1):
            margin = max(12.0, 0.6 * max(width, height))
            left, top = max(0, int(x - margin)), max(0, int(y - margin))
            right, bottom = min(source.width, int(x + width + margin)), min(source.height, int(y + height + margin))
            crop = source.crop((left, top, right, bottom))
            scale = min(150 / max(crop.width, 1), 130 / max(crop.height, 1))
            resized = crop.resize((max(1, int(crop.width * scale)), max(1, int(crop.height * scale))))
            tile = Image.new("RGB", (160, 160), (24, 24, 24))
            tile.paste(resized, ((160 - resized.width) // 2, 24 + (130 - resized.height) // 2))
            draw = ImageDraw.Draw(tile)
            draw.text((5, 5), f"candidate {candidate_id}", fill=(255, 255, 0), font=font)
            tiles.append(tile)
        columns = 4
        rows_count = (len(tiles) + columns - 1) // columns
        header = 40
        sheet = Image.new("RGB", (columns * 160, header + rows_count * 160), (15, 15, 15))
        draw = ImageDraw.Draw(sheet)
        draw.text((5, 4), identifier, fill="white", font=font)
        draw.text((5, 20), public[identifier]["goal_language"], fill="white", font=font)
        for index, tile in enumerate(tiles):
            sheet.paste(tile, ((index % columns) * 160, header + (index // columns) * 160))
        sheet.save(args.output_dir / f"{identifier}.jpg", quality=95)
    print(json.dumps({"episodes": len(predictions), "output": str(args.output_dir)}))


if __name__ == "__main__":
    main()
