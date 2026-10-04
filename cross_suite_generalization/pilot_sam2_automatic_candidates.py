"""Audit task-agnostic compact SAM 2 masks on frozen first frames."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="facebook/sam2.1-hiera-small")
    parser.add_argument("--min-area-fraction", type=float, default=0.002)
    parser.add_argument("--max-area-fraction", type=float, default=0.08)
    return parser.parse_args()


def unique_goal_jobs(jobs: list[dict]) -> list[dict]:
    chosen = {}
    for job in jobs:
        goal = str(job["goal_language"])
        chosen.setdefault(goal, {"sidecar": job["sidecar"], "goal_language": goal})
    return list(chosen.values())


def main() -> None:
    args = arguments()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    jobs = unique_goal_jobs(manifest["jobs"])
    generator = SAM2AutomaticMaskGenerator.from_pretrained(
        args.model,
        points_per_side=24,
        pred_iou_thresh=0.72,
        stability_score_thresh=0.82,
        min_mask_region_area=20,
    )
    records = []
    for goal_index, job in enumerate(jobs):
        with np.load(job["sidecar"], mmap_mode="r") as sidecar:
            image = np.asarray(sidecar["agent_images"][0], dtype=np.uint8)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            generated = generator.generate(image)
        candidates = []
        for item in generated:
            area_fraction = float(item["area"] / (image.shape[0] * image.shape[1]))
            if not args.min_area_fraction <= area_fraction <= args.max_area_fraction:
                continue
            candidates.append({
                "bbox_xywh": [float(value) for value in item["bbox"]],
                "area_fraction": area_fraction,
                "predicted_iou": float(item["predicted_iou"]),
                "stability_score": float(item["stability_score"]),
                "segmentation": np.asarray(item["segmentation"], dtype=bool),
            })
        candidates.sort(key=lambda item: (-item["predicted_iou"], -item["stability_score"]))
        canvas = image.astype(np.float32).copy()
        rng = np.random.default_rng(20260913 + goal_index)
        colors = rng.integers(40, 256, size=(len(candidates), 3))
        for candidate_index, (candidate, color) in enumerate(zip(candidates, colors)):
            mask = candidate.pop("segmentation")
            canvas[mask] = 0.6 * canvas[mask] + 0.4 * color
        rendered = Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8))
        draw = ImageDraw.Draw(rendered)
        draw.rectangle((0, 0, rendered.width, 28), fill=(0, 0, 0))
        draw.text((3, 3), job["goal_language"][:90], fill=(255, 255, 255))
        for candidate_index, (candidate, color) in enumerate(zip(candidates, colors)):
            x, y, width, height = candidate["bbox_xywh"]
            draw.rectangle((x, y, x + width, y + height), outline=tuple(int(v) for v in color), width=2)
            draw.text((x + 2, max(30, y + 2)), str(candidate_index), fill=tuple(int(v) for v in color))
        rendered.save(args.output_dir / f"goal{goal_index:02d}_candidates.png")
        records.append({
            "goal_index": goal_index,
            "goal_language": job["goal_language"],
            "num_raw_masks": len(generated),
            "num_compact_candidates": len(candidates),
            "candidates": candidates,
        })
    result = {
        "schema_version": 1,
        "model": args.model,
        "feature_firewall": ["agent RGB first frame"],
        "candidate_filter": {
            "min_area_fraction": args.min_area_fraction,
            "max_area_fraction": args.max_area_fraction,
        },
        "records": records,
    }
    (args.output_dir / "candidates.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps({
        "model": args.model,
        "images": len(records),
        "raw_masks": sum(row["num_raw_masks"] for row in records),
        "compact_candidates": sum(row["num_compact_candidates"] for row in records),
    }))


if __name__ == "__main__":
    main()
