#!/usr/bin/env python3
"""Render a post-hoc review sheet for frozen wrist candidate selections."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


COLORS = ((255, 70, 70), (70, 210, 255), (255, 210, 60))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--private-map", type=Path, required=True)
    parser.add_argument("--collection-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    scores = json.loads(args.scores.read_text(encoding="utf-8"))["records"]
    identities = {
        row["anonymous_id"]: row["source_identity"]
        for row in json.loads(args.private_map.read_text(encoding="utf-8"))["records"]
    }
    outcomes = {
        row["identity"]: row["success"]
        for row in json.loads(args.collection_audit.read_text(encoding="utf-8"))["records"]
    }
    tile_w, tile_h = 300, 280
    sheet = Image.new("RGB", (tile_w * 3, tile_h * 4), "white")
    font = ImageFont.load_default()
    for index, score in enumerate(scores):
        identifier = score["anonymous_id"]
        identity = identities[identifier]
        rgb_path = args.candidate_dir / f"{identifier}_close_rgb.jpg"
        mask_path = args.candidate_dir / f"{identifier}_initial_masks.npz"
        if rgb_path.exists():
            image = Image.open(rgb_path).convert("RGB")
        else:
            image = Image.new("RGB", (224, 224), (235, 235, 235))
        array = np.asarray(image).copy()
        selected = [int(value) for value in score.get("candidate_set", [])]
        if mask_path.exists() and selected:
            with np.load(mask_path, allow_pickle=False) as archive:
                masks = np.asarray(archive["masks"], dtype=bool)
                boxes = np.asarray(archive["boxes_xywh"], dtype=float)
            for color_index, candidate_id in enumerate(selected):
                mask = masks[candidate_id - 1]
                color = np.asarray(COLORS[color_index % len(COLORS)])
                array[mask] = (0.55 * array[mask] + 0.45 * color).astype(np.uint8)
                x, y, width, height = boxes[candidate_id - 1]
                annotated = Image.fromarray(array)
                draw = ImageDraw.Draw(annotated)
                draw.rectangle((x, y, x + width, y + height), outline=tuple(color), width=2)
                draw.text((x + 2, y + 2), str(candidate_id), fill=tuple(color), font=font)
                array = np.asarray(annotated).copy()
        image = Image.fromarray(array).resize((224, 224))
        tile = Image.new("RGB", (tile_w, tile_h), "white")
        tile.paste(image, ((tile_w - 224) // 2, 44))
        draw = ImageDraw.Draw(tile)
        probability = score.get("controlled_object_candidate_probability")
        probability_text = "none" if probability is None else f"{probability:.3f}"
        outcome_text = "success" if outcomes[identity] else "failure"
        draw.text((5, 4), identity, fill="black", font=font)
        draw.text(
            (5, 20),
            f"{outcome_text} | {score['state']} | ids={selected} | p={probability_text} | gate={bool(score.get('passes_frozen_candidate_gate'))}",
            fill="black",
            font=font,
        )
        sheet.paste(tile, ((index % 3) * tile_w, (index // 3) * tile_h))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.output, quality=95)
    print(args.output)


if __name__ == "__main__":
    main()
