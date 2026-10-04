"""Render dense, anonymous sheets around every registered close event."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
from PIL import Image, ImageDraw


OFFSETS = (-10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 40)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-dir", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = json.loads(args.index.read_text(encoding="utf-8"))["records"]
    rendered = []
    for record in records:
        path = args.video_dir / record["video"]
        capture = cv2.VideoCapture(str(path))
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        for event_number, close_frame in enumerate(record["close_frames"]):
            indices = sorted(set(max(0, min(count - 1, close_frame + offset)) for offset in OFFSETS))
            columns, cell_width, cell_height, label_height = 4, 336, 168, 20
            rows = (len(indices) + columns - 1) // columns
            canvas = Image.new("RGB", (columns * cell_width, rows * (cell_height + label_height)), "white")
            draw = ImageDraw.Draw(canvas)
            for slot, index in enumerate(indices):
                capture.set(cv2.CAP_PROP_POS_FRAMES, index)
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError(f"could not read frame {index} from {path}")
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                image = image.resize((cell_width, cell_height))
                x = (slot % columns) * cell_width
                y = (slot // columns) * (cell_height + label_height)
                canvas.paste(image, (x, y + label_height))
                marker = " CLOSE" if index == close_frame else ""
                draw.text((x + 4, y + 3), f"frame {index} ({index-close_frame:+d}){marker}", fill="black")
            output = args.output_dir / f"{record['anonymous_id']}_close{event_number:02d}_f{close_frame:04d}.jpg"
            canvas.save(output, quality=94)
            rendered.append({
                "anonymous_id": record["anonymous_id"],
                "event_number": event_number,
                "close_frame": close_frame,
                "sheet": output.name,
                "sampled_frames": indices,
            })
        capture.release()
    (args.output_dir / "index.json").write_text(
        json.dumps({"schema_version": 1, "offsets": OFFSETS, "records": rendered},
                   ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"event_sheets": len(rendered)}, indent=2))


if __name__ == "__main__":
    main()
