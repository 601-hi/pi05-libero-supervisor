#!/usr/bin/env python3
"""Render outcome- and language-blind wrist candidate review sheets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


COLORS = [(0, 215, 255), (255, 120, 30), (80, 220, 80), (220, 80, 220),
          (40, 130, 255), (255, 80, 80), (160, 220, 40), (180, 120, 255)]


def valid_point(value):
    if value is None or len(value) != 2:
        return None
    return int(round(value[0])), int(round(value[1]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anonymous-sidecars", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    sidecars = {r["anonymous_id"]: r["sidecar"] for r in json.loads(args.anonymous_sidecars.read_text(encoding="utf-8"))["records"]}
    predictions = json.loads(args.predictions.read_text(encoding="utf-8"))["records"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    template = []
    for record in predictions:
        identifier = record["anonymous_id"]
        close = int(record["close_frame"])
        with np.load(sidecars[identifier], allow_pickle=False) as data:
            wrist = np.asarray(data["wrist_images"], dtype=np.uint8)
        with np.load(args.predictions.parent / f"{identifier}_initial_masks.npz", allow_pickle=False) as data:
            masks = np.asarray(data["masks"], dtype=bool)
        tracked_path = args.predictions.parent / f"{identifier}_tracked_masks.npz"
        tracked = np.load(tracked_path, allow_pickle=False) if tracked_path.exists() else None
        offsets = [-15, -10, -5, 0, 5, 10, 20]
        panels = []
        for offset in offsets:
            frame = min(max(0, close + offset), len(wrist) - 1)
            panel = cv2.cvtColor(wrist[frame], cv2.COLOR_RGB2BGR)
            if offset == 0:
                overlay = panel.copy()
                for candidate_index, mask in enumerate(masks):
                    color = COLORS[candidate_index % len(COLORS)]
                    overlay[mask] = np.asarray(color, dtype=np.uint8)
                panel = cv2.addWeighted(overlay, .35, panel, .65, 0)
            relative_key = "pre_centroids_xy" if offset <= 0 else "post_centroids_xy"
            relative_frames = record["pre_relative_frames"] if offset <= 0 else record["post_relative_frames"]
            if offset in relative_frames:
                track_index = relative_frames.index(offset)
                for raw_id, sequence in record[relative_key].items():
                    if track_index >= len(sequence):
                        continue
                    point = valid_point(sequence[track_index])
                    if point is None:
                        continue
                    color = COLORS[(int(raw_id) - 1) % len(COLORS)]
                    if tracked is not None:
                        phase = "pre" if offset <= 0 else "post"
                        key = f"{phase}_{raw_id}"
                        if key in tracked and track_index < len(tracked[key]):
                            mask = np.asarray(tracked[key][track_index], dtype=np.uint8)
                            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            cv2.drawContours(panel, contours, -1, color, 1, cv2.LINE_AA)
                    cv2.circle(panel, point, 4, color, -1, cv2.LINE_AA)
                    cv2.putText(panel, str(raw_id), (point[0] + 4, point[1] - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, .42, color, 1, cv2.LINE_AA)
            title = np.full((26, 224, 3), 245, dtype=np.uint8)
            cv2.putText(title, f"frame {frame} ({offset:+d})" + (" CLOSE" if offset == 0 else ""),
                        (4, 18), cv2.FONT_HERSHEY_SIMPLEX, .44, (0, 0, 0), 1, cv2.LINE_AA)
            panels.append(np.concatenate([title, panel], axis=0))
        if tracked is not None:
            tracked.close()
        canvas = np.full((2 * 250, 4 * 224, 3), 255, dtype=np.uint8)
        for index, panel in enumerate(panels):
            row, column = divmod(index, 4)
            canvas[row * 250:(row + 1) * 250, column * 224:(column + 1) * 224] = panel
        output = args.output_dir / f"{identifier}_wrist_physical.jpg"
        cv2.imwrite(str(output), canvas, [cv2.IMWRITE_JPEG_QUALITY, 93])
        template.append({
            "anonymous_id": identifier,
            "close_frame": close,
            "acceptable_controlled_candidate_ids": [],
            "controlled_object_observable": None,
            "confidence": None,
            "notes": "Multiple IDs are allowed only for overlapping masks of the same physically controlled object.",
        })
    (args.output_dir / "physical_annotations.template.json").write_text(
        json.dumps({
            "schema_version": 1,
            "annotation_pass": "physical_control_blind",
            "forbidden_information": ["task language", "episode outcome", "reward", "semantic target", "model ranking"],
            "records": template,
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"episodes": len(template), "sheets": len(template)}))


if __name__ == "__main__":
    main()
