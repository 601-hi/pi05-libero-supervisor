#!/usr/bin/env python3
"""Render label-neutral contact sheets for causal identity holdout review.

No predicted/locked candidate is highlighted.  All candidates use the same
appearance so a reviewer can assign manipulated-object identity before the
prediction is revealed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


PALETTE = (
    (0, 255, 255), (255, 215, 0), (255, 105, 180), (50, 205, 50),
    (255, 140, 0), (138, 43, 226), (0, 191, 255), (255, 99, 71),
    (127, 255, 0), (238, 130, 238), (64, 224, 208), (255, 255, 255),
)


def candidate_color(candidate_id: int) -> tuple[int, int, int]:
    """Deterministic display color; independent of the predicted identity."""
    return PALETTE[(candidate_id - 1) % len(PALETTE)]


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--public-manifest", type=Path, required=True)
    parser.add_argument("--private-map", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--wave", type=int, default=1)
    parser.add_argument("--panels", type=int, default=9)
    return parser.parse_args()


def text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], value: str, fill=(255, 255, 255)) -> None:
    x, y = xy
    box = draw.textbbox((x, y), value, font=ImageFont.load_default())
    draw.rectangle((box[0] - 2, box[1] - 1, box[2] + 2, box[3] + 1), fill=(0, 0, 0))
    draw.text((x, y), value, fill=fill, font=ImageFont.load_default())


def main() -> None:
    args = arguments()
    public = json.loads(args.public_manifest.read_text(encoding="utf-8"))
    private = json.loads(args.private_map.read_text(encoding="utf-8"))
    predictions = json.loads(args.predictions.read_text(encoding="utf-8"))
    public_by_id = {row["anonymous_id"]: row for row in public["records"]}
    private_by_id = {row["anonymous_id"]: row for row in private["records"]}
    rows = [row for row in predictions["records"] if public_by_id[row["anonymous_id"]]["evaluation_wave"] == args.wave]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    template = []
    for row in rows:
        anonymous_id = row["anonymous_id"]
        sidecar = Path(private_by_id[anonymous_id]["original_sidecar"])
        with np.load(sidecar, mmap_mode="r") as data:
            frames = np.asarray(data["agent_images"], dtype=np.uint8)
        indices = np.linspace(0, len(frames) - 1, args.panels, dtype=int).tolist()
        panels = []
        tracks = {int(key): value for key, value in row["centroids_xy"].items()}
        for frame_index in indices:
            panel = Image.fromarray(frames[frame_index]).convert("RGB")
            draw = ImageDraw.Draw(panel)
            if frame_index == 0:
                for candidate_id, (x, y, width, height) in enumerate(row["initial_boxes_xywh"], start=1):
                    color = candidate_color(candidate_id)
                    draw.rectangle((x, y, x + width, y + height), outline=color, width=2)
                    text(draw, (int(x) + 2, int(y) + 2), str(candidate_id), fill=color)
            for candidate_id, points in tracks.items():
                color = candidate_color(candidate_id)
                history = [tuple(map(int, point)) for point in points[: frame_index + 1] if point is not None]
                if len(history) >= 2:
                    draw.line(history, fill=color, width=2)
                point = points[frame_index]
                if point is None:
                    continue
                x, y = map(int, point)
                draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color)
                text(draw, (x + 4, y - 7), str(candidate_id), fill=color)
            text(draw, (5, 5), f"frame {frame_index}")
            panels.append(panel)

        width, height = panels[0].size
        header_height = 44
        columns = 3
        rows_count = (len(panels) + columns - 1) // columns
        sheet = Image.new("RGB", (width * columns, height * rows_count + header_height), (20, 20, 20))
        sheet_draw = ImageDraw.Draw(sheet)
        goal = public_by_id[anonymous_id]["goal_language"]
        text(sheet_draw, (6, 5), anonymous_id)
        text(sheet_draw, (6, 22), goal)
        for panel_index, panel in enumerate(panels):
            column, row_index = panel_index % columns, panel_index // columns
            sheet.paste(panel, (column * width, header_height + row_index * height))
        sheet.save(args.output_dir / f"{anonymous_id}.jpg", quality=92)
        template.append({
            "anonymous_id": anonymous_id,
            "acceptable_manipulated_candidate_ids": [],
            "identity_observable": None,
            "first_observable_frame": None,
            "confidence": None,
            "notes": "Use multiple IDs only when overlapping masks represent the same manipulated object.",
        })

    (args.output_dir / "blind_annotations.template.json").write_text(
        json.dumps({"schema_version": 1, "records": template}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"episodes": len(rows), "sheets": len(rows), "panels_per_sheet": args.panels}))


if __name__ == "__main__":
    main()
