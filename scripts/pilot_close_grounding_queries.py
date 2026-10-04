#!/usr/bin/env python3
"""Compare state-aware grounding queries for an articulated close goal."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor


QUERIES = (
    "top drawer",
    "open drawer",
    "open top drawer",
    "open cabinet drawer",
    "drawer opening",
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sidecar", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--model", default="IDEA-Research/grounding-dino-base")
    ap.add_argument("--frames", type=int, nargs="+", default=[0, 200, 399])
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(args.model).to(device).eval()
    records = []
    with np.load(args.sidecar, mmap_mode="r") as data:
        for frame in args.frames:
            image = Image.fromarray(np.asarray(data["agent_images"][frame], dtype=np.uint8)).convert("RGB")
            canvas = image.copy()
            draw = ImageDraw.Draw(canvas)
            for query_index, query in enumerate(QUERIES):
                inputs = processor(images=image, text=query + ".", return_tensors="pt").to(device)
                with torch.inference_mode():
                    output = model(**inputs)
                result = processor.post_process_grounded_object_detection(
                    output, inputs.input_ids, threshold=0.12, text_threshold=0.10,
                    target_sizes=[image.size[::-1]],
                )[0]
                labels = result.get("text_labels", result["labels"])
                for box, score, label in zip(result["boxes"], result["scores"], labels):
                    values = [float(v) for v in box]
                    item = {
                        "frame": frame, "query": query, "label": str(label),
                        "score": float(score), "box_xyxy": values,
                    }
                    records.append(item)
                    if float(score) >= 0.20:
                        color = ((255, 50, 50), (50, 255, 50), (50, 150, 255),
                                 (255, 220, 50), (220, 50, 255))[query_index]
                        draw.rectangle(values, outline=color, width=2)
                        draw.text((values[0] + 1, max(1, values[1] + 1)),
                                  f"{query_index}:{score:.2f}", fill=color)
            canvas.save(args.output_dir / f"frame{frame:04d}.png")
    payload = {
        "schema_version": 1,
        "feature_firewall": ["fixed-camera RGB", "state-aware relation query"],
        "query_legend": {str(i): q for i, q in enumerate(QUERIES)},
        "records": records,
    }
    (args.output_dir / "predictions.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"device": device, "frames": args.frames, "predictions": len(records)}))


if __name__ == "__main__":
    main()
