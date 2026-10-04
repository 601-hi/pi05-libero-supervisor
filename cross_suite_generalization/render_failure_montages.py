"""Render evenly sampled fixed/wrist contact sheets for selected episode sidecars."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sidecar", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=10)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for path in args.sidecar:
        with np.load(path, mmap_mode="r") as archive:
            length = len(archive["action_indices"])
            indices = np.linspace(0, length - 1, args.frames, dtype=int)
            fixed = np.asarray(archive["agent_images"][indices], dtype=np.uint8)
            wrist = np.asarray(archive["wrist_images"][indices], dtype=np.uint8)
            actions = np.asarray(archive["action_indices"])[indices]
        height, width = fixed.shape[1:3]
        canvas = Image.new("RGB", (args.frames * width, 2 * height + 24), "white")
        draw = ImageDraw.Draw(canvas)
        for column, (fixed_frame, wrist_frame, action) in enumerate(zip(fixed, wrist, actions)):
            x = column * width
            canvas.paste(Image.fromarray(fixed_frame), (x, 24))
            canvas.paste(Image.fromarray(wrist_frame), (x, 24 + height))
            draw.text((x + 3, 4), f"a{int(action)}", fill="black")
        output = args.output_dir / f"{path.stem}.png"
        canvas.save(output)
        print(output)


if __name__ == "__main__":
    main()
