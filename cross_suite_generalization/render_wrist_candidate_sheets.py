#!/usr/bin/env python3
"""Render close-frame wrist candidates without exposing temporal predictions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


COLORS = [(0,200,220),(245,190,20),(230,60,160),(60,200,80),(240,130,20),(130,60,230),(20,160,220),(230,80,60)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = json.loads(args.predictions.read_text(encoding="utf-8"))["records"]
    for row in rows:
        identifier = row["anonymous_id"]
        rgb = np.asarray(Image.open(args.input_dir / f"{identifier}_close_rgb.jpg").convert("RGB"))
        masks = np.load(args.input_dir / f"{identifier}_initial_masks.npz")["masks"].astype(bool)
        tile, columns = 180, 4
        canvas = Image.new("RGB", (tile * columns, tile * 2), "black")
        draw = ImageDraw.Draw(canvas)
        for index, mask in enumerate(masks):
            points = np.argwhere(mask)
            if not len(points):
                continue
            low, high = points.min(0), points.max(0)
            pad = 8
            y0,x0=np.maximum(low-pad,0); y1,x1=np.minimum(high+pad+1,rgb.shape[:2])
            crop=rgb[y0:y1,x0:x1].copy(); local=mask[y0:y1,x0:x1]
            color=np.asarray(COLORS[index % len(COLORS)],dtype=float)
            crop[local]=np.clip(0.45*crop[local]+0.55*color,0,255).astype(np.uint8)
            image=Image.fromarray(crop); image.thumbnail((tile-12,tile-28))
            x=(index%columns)*tile+(tile-image.width)//2; y=(index//columns)*tile+24
            canvas.paste(image,(x,y)); draw.text(((index%columns)*tile+5,(index//columns)*tile+5),f"candidate {index+1}",fill=COLORS[index%len(COLORS)])
        canvas.save(args.output_dir/f"{identifier}_wrist_mask_sheet.jpg",quality=95)
    print(json.dumps({"episodes":len(rows),"output":str(args.output_dir)}))


if __name__ == "__main__":
    main()

