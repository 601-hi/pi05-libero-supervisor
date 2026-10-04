#!/usr/bin/env python3
"""Render close-event RGB frames for simulator-independent hand-eye annotation."""
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
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    public = {row["anonymous_id"]: row for row in json.loads(args.public.read_text(encoding="utf-8"))["records"]}
    private = {row["anonymous_id"]: row for row in json.loads(args.private.read_text(encoding="utf-8"))["records"]}
    features = json.loads(args.features.read_text(encoding="utf-8"))["records"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    template = []
    font = ImageFont.load_default()
    for row in features:
        anonymous_id = row["anonymous_id"]
        with np.load(private[anonymous_id]["original_sidecar"], mmap_mode="r") as data:
            frames = np.asarray(data["agent_images"], dtype=np.uint8)
        # The first close is the primary grasp attempt. Later attempts remain in
        # the metadata for future retry-state annotations.
        frame_index = row["close_frames"][0]
        image = Image.fromarray(frames[frame_index]).convert("RGB").resize((448, 448))
        draw = ImageDraw.Draw(image)
        for pixel in range(0, 225, 16):
            coordinate = pixel * 2
            draw.line((coordinate, 0, coordinate, 448), fill=(255, 255, 0), width=1)
            draw.line((0, coordinate, 448, coordinate), fill=(255, 255, 0), width=1)
            draw.text((coordinate + 2, 2), str(pixel), fill=(255, 255, 0), font=font)
            draw.text((2, coordinate + 2), str(pixel), fill=(255, 255, 0), font=font)
        image.save(args.output_dir / f"{anonymous_id}_close{frame_index:04d}.png")
        template.append({
            "anonymous_id": anonymous_id,
            "close_frame": frame_index,
            "gripper_center_xy": None,
            "visibility": None,
            "confidence": None,
            "goal_language": public[anonymous_id]["goal_language"],
        })
    (args.output_dir / "gripper_pixel_annotations.template.json").write_text(
        json.dumps({"schema_version": 1, "records": template}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"frames": len(template), "output": str(args.output_dir)}))


if __name__ == "__main__":
    main()
