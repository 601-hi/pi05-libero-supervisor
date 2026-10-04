"""Render conservative dual-view review sheets for possible-control events."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


COLORS = [(255, 55, 55), (55, 235, 90), (50, 150, 255), (255, 190, 40)]


def overlay(frame: np.ndarray, masks: list[np.ndarray], labels: list[str]) -> Image.Image:
    canvas = frame.astype(float).copy()
    for index, mask in enumerate(masks):
        color = np.asarray(COLORS[index % len(COLORS)], dtype=float)
        mask = np.asarray(mask, dtype=bool)
        canvas[mask] = .45 * canvas[mask] + .55 * color
    image = Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(image)
    for index, label in enumerate(labels):
        draw.rectangle((2, 2 + index * 14, 112, 15 + index * 14), fill=(0, 0, 0))
        draw.text((4, 2 + index * 14), label, fill=COLORS[index % len(COLORS)])
    return image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--wrist-predictions", type=Path, required=True)
    parser.add_argument("--wrist-scores", type=Path, required=True)
    parser.add_argument("--wrist-dir", type=Path, required=True)
    parser.add_argument("--fixed-motion", type=Path, required=True)
    parser.add_argument("--fixed-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--include-event-ids", default="",
                        help="Comma-separated diagnostic override; does not change scoring.")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    sidecars = {episode["anonymous_id"]: episode["sidecar"] for episode in manifest["episodes"]}
    wp = {r["event_id"]: r for r in json.loads(args.wrist_predictions.read_text(encoding="utf-8"))["records"]}
    ws = {r["event_id"]: r for r in json.loads(args.wrist_scores.read_text(encoding="utf-8"))["records"]}
    fm = {r["event_id"]: r for r in json.loads(args.fixed_motion.read_text(encoding="utf-8"))["records"]}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    forced = {value for value in args.include_event_ids.split(",") if value}
    rendered = []
    for event_id, score in ws.items():
        motion = fm[event_id]
        if event_id not in forced and (
            not score["passes_frozen_candidate_gate"] or not motion["sustained_candidate_ids"]
        ):
            continue
        prediction, close = wp[event_id], int(score["close_frame"])
        with np.load(sidecars[score["anonymous_id"]], allow_pickle=False) as data:
            wrist_rgb = np.asarray(data["wrist_images"], dtype=np.uint8)
            fixed_rgb = np.asarray(data["agent_images"], dtype=np.uint8)
        with np.load(args.wrist_dir / f"{event_id}_tracked_masks.npz", allow_pickle=False) as data:
            wrist_masks = {key: np.asarray(data[key]) for key in data.files}
        with np.load(args.fixed_dir / f"{event_id}_tracked_masks.npz", allow_pickle=False) as data:
            fixed_masks = {key: np.asarray(data[key]) for key in data.files}
        wrist_ids = [str(value) for value in score["candidate_set"]]
        if not wrist_ids or not motion["sustained_candidate_ids"]:
            continue
        fixed_ids = motion["sustained_candidate_ids"][:4]
        offsets = [0, min(10, len(prediction["post_centroids_xy"][wrist_ids[0]]) - 1),
                   min(25, len(prediction["post_centroids_xy"][wrist_ids[0]]) - 1)]
        tiles = []
        for camera, frames, masks, identifiers in (
            ("WRIST", wrist_rgb, wrist_masks, wrist_ids),
            ("FIXED", fixed_rgb, fixed_masks, fixed_ids),
        ):
            row = []
            for offset in offsets:
                selected_masks = [masks[f"post_{identifier}"][offset] for identifier in identifiers]
                image = overlay(frames[close + offset], selected_masks,
                                [f"{camera} id={value}" for value in identifiers])
                draw = ImageDraw.Draw(image)
                draw.text((150, 4), f"step {close + offset}", fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))
                row.append(image)
            tiles.append(row)
        width, height = tiles[0][0].size
        sheet = Image.new("RGB", (width * 3, height * 2), (20, 20, 20))
        for y, row in enumerate(tiles):
            for x, image in enumerate(row):
                sheet.paste(image, (x * width, y * height))
        path = args.output_dir / f"{event_id}.png"
        sheet.save(path)
        rendered.append(str(path))
    print(json.dumps({"rendered": len(rendered), "paths": rendered}, indent=2))


if __name__ == "__main__":
    main()
