#!/usr/bin/env python3
"""Blind open-vs-closed state query contrast on frozen RGB frames."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--model", default="IDEA-Research/grounding-dino-base")
    args = ap.parse_args()
    jobs = json.loads(args.jobs.read_text(encoding="utf-8"))["jobs"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(args.model).to(device).eval()
    rows = []
    queries = ("open top drawer", "closed top drawer")
    for job in jobs:
        with np.load(job["sidecar"], mmap_mode="r") as data:
            for frame in job["frames"]:
                image = Image.fromarray(np.asarray(data["agent_images"][frame], dtype=np.uint8)).convert("RGB")
                states = {}
                for query in queries:
                    inputs = processor(images=image, text=query + ".", return_tensors="pt").to(device)
                    with torch.inference_mode():
                        raw = model(**inputs)
                    result = processor.post_process_grounded_object_detection(
                        raw, inputs.input_ids, threshold=0.08, text_threshold=0.06,
                        target_sizes=[image.size[::-1]],
                    )[0]
                    candidates = sorted((
                        {"score": float(s), "box_xyxy": [float(x) for x in b]}
                        for b, s in zip(result["boxes"], result["scores"])
                    ), key=lambda item: item["score"], reverse=True)
                    states[query] = candidates[:8]
                rows.append({"identity": job["identity"], "frame": frame, "states": states})
    result = {
        "schema_version": 1,
        "feature_firewall": ["fixed-camera RGB", "paired open/closed language queries"],
        "forbidden_inputs": ["reward", "outcome", "simulator state"],
        "records": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"device": device, "images": len(rows), "output": str(args.output)}))


if __name__ == "__main__":
    main()
