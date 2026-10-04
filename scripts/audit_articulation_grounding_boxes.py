#!/usr/bin/env python3
"""Audit online DINO first-frame boxes against frozen development boxes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor


def iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0., x1 - x0) * max(0., y1 - y0)
    area_a = max(0., a[2] - a[0]) * max(0., a[3] - a[1])
    area_b = max(0., b[2] - b[0]) * max(0., b[3] - b[1])
    return intersection / max(area_a + area_b - intersection, 1e-12)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="IDEA-Research/grounding-dino-base")
    parser.add_argument("--query", default="open top drawer")
    parser.add_argument("--minimum-iou", type=float, default=.90)
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(args.model).to(device).eval()
    records = []
    for path in args.predictions:
        for record in json.loads(path.read_text(encoding="utf-8"))["records"]:
            with np.load(record["sidecar"], mmap_mode="r") as data:
                image = Image.fromarray(np.asarray(data["agent_images"][0], np.uint8))
            inputs = processor(images=image, text=args.query + ".",
                               return_tensors="pt").to(device)
            with torch.inference_mode(): output = model(**inputs)
            result = processor.post_process_grounded_object_detection(
                output, inputs.input_ids, threshold=.12, text_threshold=.10,
                target_sizes=[image.size[::-1]])[0]
            index = int(torch.argmax(result["scores"]).item())
            box = result["boxes"][index].detach().cpu().numpy().tolist()
            frozen = record["anchor_box_xyxy"]
            overlap = float(iou(box, frozen))
            item = {"identity": record["identity"], "score": float(result["scores"][index]),
                    "box_xyxy": box, "frozen_box_xyxy": frozen,
                    "iou": overlap, "passed": overlap >= args.minimum_iou}
            records.append(item); print(json.dumps(item), flush=True)
    payload = {"schema_version": 1,
               "status": "passed" if records and all(x["passed"] for x in records) else "failed",
               "minimum_iou": args.minimum_iou, "records": records}
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    raise SystemExit(0 if payload["status"] == "passed" else 2)


if __name__ == "__main__": main()
