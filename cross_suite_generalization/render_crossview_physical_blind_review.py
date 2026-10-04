#!/usr/bin/env python3
"""Combine label-blind fixed-view motion and wrist-mask evidence sheets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def labelled_panel(image: np.ndarray, label: str, width: int) -> np.ndarray:
    scale = width / image.shape[1]
    resized = cv2.resize(image, (width, int(round(image.shape[0] * scale))))
    header = np.full((34, width, 3), 245, dtype=np.uint8)
    cv2.putText(header, label, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, .58, (0, 0, 0), 1, cv2.LINE_AA)
    return np.concatenate([header, resized], axis=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--fixed-sheets", type=Path, required=True)
    parser.add_argument("--wrist-sheets", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--width", type=int, default=1120)
    args = parser.parse_args()

    records = json.loads(args.predictions.read_text(encoding="utf-8"))["records"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    template = []
    for record in records:
        identifier = record["anonymous_id"]
        fixed_matches = sorted(args.fixed_sheets.glob(f"{identifier}_close00_*_motion.jpg"))
        if len(fixed_matches) != 1:
            raise RuntimeError(f"Expected one first-close fixed sheet for {identifier}, got {fixed_matches}")
        wrist_path = args.wrist_sheets / f"{identifier}_wrist_physical.jpg"
        fixed = cv2.imread(str(fixed_matches[0]))
        wrist = cv2.imread(str(wrist_path))
        if fixed is None or wrist is None:
            raise RuntimeError(f"Unreadable evidence image for {identifier}")
        top = labelled_panel(fixed, "FIXED VIEW: stationary-background compensated motion groups", args.width)
        bottom = labelled_panel(wrist, "WRIST VIEW: bidirectional candidate masks around first close", args.width)
        separator = np.full((8, args.width, 3), 255, dtype=np.uint8)
        canvas = np.concatenate([top, separator, bottom], axis=0)
        output = args.output_dir / f"{identifier}_crossview_physical.jpg"
        if not cv2.imwrite(str(output), canvas, [cv2.IMWRITE_JPEG_QUALITY, 94]):
            raise RuntimeError(f"Failed to write {output}")
        template.append({
            "anonymous_id": identifier,
            "close_frame": int(record["close_frame"]),
            "acceptable_controlled_candidate_ids": [],
            "controlled_object_observable": None,
            "contact_or_control_evidence": "unknown",
            "confidence": None,
            "notes": "Do not infer task semantics or episode outcome. Multiple IDs require the same physical object.",
        })

    manifest = {
        "schema_version": 1,
        "annotation_pass": "crossview_physical_control_blind",
        "allowed_information": ["fixed-view background-compensated motion", "wrist candidate masks", "first-close timing"],
        "forbidden_information": ["task language", "episode outcome", "reward", "semantic target", "old target-mask labels", "model ranking"],
        "records": template,
    }
    (args.output_dir / "physical_annotations.template.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"episodes": len(template), "sheets": len(template)}))


if __name__ == "__main__":
    main()
