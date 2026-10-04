#!/usr/bin/env python3
"""Score persistence of an articulated mechanism's counter-state.

For a Close goal the counter-state query is, for example, "open top drawer".
The first-frame candidate anchors identity. Later detections are associated by
IoU with that anchor; simulator state, reward and task outcome are forbidden.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor


def iou(a: np.ndarray, b: np.ndarray) -> float:
    left, top = np.maximum(a[:2], b[:2])
    right, bottom = np.minimum(a[2:], b[2:])
    inter = max(0.0, right - left) * max(0.0, bottom - top)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return float(inter / (area_a + area_b - inter + 1e-9))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sidecar", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--query", required=True)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--model", default="IDEA-Research/grounding-dino-base")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(args.model).to(device).eval()
    records: list[dict] = []
    anchor: np.ndarray | None = None

    with np.load(args.sidecar, mmap_mode="r") as data:
        count = len(data["agent_images"])
        frame_indices = list(range(0, count, args.stride))
        if frame_indices[-1] != count - 1:
            frame_indices.append(count - 1)
        for frame in frame_indices:
            image = Image.fromarray(np.asarray(data["agent_images"][frame], dtype=np.uint8)).convert("RGB")
            inputs = processor(images=image, text=args.query + ".", return_tensors="pt").to(device)
            with torch.inference_mode():
                raw = model(**inputs)
            result = processor.post_process_grounded_object_detection(
                raw, inputs.input_ids, threshold=0.10, text_threshold=0.08,
                target_sizes=[image.size[::-1]],
            )[0]
            candidates = [
                {"score": float(score), "box_xyxy": [float(x) for x in box]}
                for box, score in zip(result["boxes"], result["scores"])
            ]
            if anchor is None:
                # State-aware query should rank the actual counter-state first.
                selected = max(candidates, key=lambda x: x["score"], default=None)
                if selected is None:
                    raise RuntimeError("counter-state query produced no initial candidate")
                anchor = np.asarray(selected["box_xyxy"], dtype=float)
                selected_iou = 1.0
            else:
                eligible = []
                for candidate in candidates:
                    overlap = iou(anchor, np.asarray(candidate["box_xyxy"], dtype=float))
                    if overlap >= 0.15:
                        eligible.append((overlap, candidate["score"], candidate))
                if eligible:
                    selected_iou, _, selected = max(eligible, key=lambda item: (item[0], item[1]))
                else:
                    selected, selected_iou = None, 0.0
            records.append({
                "frame": frame,
                "selected_score": selected["score"] if selected else 0.0,
                "selected_iou_to_anchor": selected_iou,
                "selected_box_xyxy": selected["box_xyxy"] if selected else None,
                "candidate_count": len(candidates),
            })

    result = {
        "schema_version": 1,
        "sidecar": str(args.sidecar),
        "query": args.query,
        "model": args.model,
        "device": device,
        "stride": args.stride,
        "feature_firewall": ["fixed-camera RGB", "counter-state query", "first-frame identity anchor"],
        "forbidden_inputs": ["reward", "episode outcome", "simulator object pose", "task id threshold"],
        "anchor_box_xyxy": anchor.tolist() if anchor is not None else None,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"frames": len(records), "anchor": result["anchor_box_xyxy"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
