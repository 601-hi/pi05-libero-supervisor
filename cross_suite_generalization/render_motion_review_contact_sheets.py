"""Render evenly sampled contact sheets from anonymous motion-review videos."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from PIL import Image, ImageDraw


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=12)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for path in sorted(args.input_dir.glob("*_motion_review.mp4")):
        capture = cv2.VideoCapture(str(path))
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = [round(i * (count - 1) / max(args.samples - 1, 1)) for i in range(args.samples)]
        images: list[Image.Image] = []
        for index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"Could not read frame {index} from {path}")
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)
            image.thumbnail((448, 224))
            images.append(image)
        capture.release()

        columns = 3
        rows = (len(images) + columns - 1) // columns
        canvas = Image.new("RGB", (448 * columns, 250 * rows), "white")
        draw = ImageDraw.Draw(canvas)
        for slot, (index, image) in enumerate(zip(indices, images)):
            x = (slot % columns) * 448
            y = (slot // columns) * 250
            canvas.paste(image, (x, y + 22))
            draw.text((x + 5, y + 4), f"frame {index}", fill="black")
        output = args.output_dir / path.name.replace("_motion_review.mp4", "_motion_sheet.jpg")
        canvas.save(output, quality=93)
    print(f"rendered={len(list(args.output_dir.glob('*_motion_sheet.jpg')))}")


if __name__ == "__main__":
    main()
