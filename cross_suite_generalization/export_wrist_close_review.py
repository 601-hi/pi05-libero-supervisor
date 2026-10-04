#!/usr/bin/env python3
"""Render outcome-free wrist frames around the first gripper-close event."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--public", type=Path, required=True)
    parser.add_argument("--private", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    public = json.loads(args.public.read_text(encoding="utf-8"))["records"]
    private = {r["anonymous_id"]: r for r in json.loads(args.private.read_text(encoding="utf-8"))["records"]}
    features = {r["anonymous_id"]: r for r in json.loads(args.features.read_text(encoding="utf-8"))["records"]}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for row in public:
        identifier = row["anonymous_id"]
        closes = features[identifier]["close_frames"]
        if not closes:
            manifest.append({"anonymous_id": identifier, "close_frame": None, "review_frames": []})
            continue
        with np.load(private[identifier]["original_sidecar"], allow_pickle=False) as data:
            frames = np.asarray(data["wrist_images"], dtype=np.uint8)
        close = int(closes[0])
        indices = sorted(set(min(close + offset, len(frames) - 1) for offset in (0, 10, 20)))
        canvas = Image.new("RGB", (224 * len(indices), 252), "white")
        draw = ImageDraw.Draw(canvas)
        for column, frame_index in enumerate(indices):
            canvas.paste(Image.fromarray(frames[frame_index]), (224 * column, 28))
            draw.text((224 * column + 4, 5), f"frame {frame_index} (+{frame_index-close})", fill="black")
        output = args.output_dir / f"{identifier}_wrist_close_review.jpg"
        canvas.save(output, quality=95)
        manifest.append({"anonymous_id": identifier, "close_frame": close, "review_frames": indices, "image": output.name})
    (args.output_dir / "manifest.json").write_text(
        json.dumps({"schema_version": 1, "records": manifest}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"episodes": len(manifest), "rendered": sum(bool(r["review_frames"]) for r in manifest)}))


if __name__ == "__main__":
    main()
