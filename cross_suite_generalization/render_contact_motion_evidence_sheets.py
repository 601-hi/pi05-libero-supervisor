#!/usr/bin/env python3
"""Render blinded close-event sheets with fixed-view candidate motion groups."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


COLORS = [(20, 210, 255), (255, 100, 30), (80, 220, 80), (220, 80, 220), (40, 130, 255)]


def point(value: object) -> tuple[int, int] | None:
    if not isinstance(value, list) or len(value) != 2 or value[0] is None or value[1] is None:
        return None
    return int(round(value[0])), int(round(value[1]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anonymous-sidecars", type=Path, required=True)
    parser.add_argument("--motion-evidence", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    sidecars = {r["anonymous_id"]: r["sidecar"] for r in json.loads(args.anonymous_sidecars.read_text(encoding="utf-8"))["records"]}
    evidence = json.loads(args.motion_evidence.read_text(encoding="utf-8"))["records"]
    predictions = {}
    for path in args.predictions:
        predictions.update({r["anonymous_id"]: r for r in json.loads(path.read_text(encoding="utf-8"))["records"]})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    offsets = [-10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 40]
    rendered = 0
    for episode in evidence:
        identifier = episode["anonymous_id"]
        with np.load(sidecars[identifier], allow_pickle=False) as data:
            fixed = np.asarray(data["agent_images"], dtype=np.uint8)
            wrist = np.asarray(data["wrist_images"], dtype=np.uint8)
        tracks = predictions[identifier]["centroids_xy"]
        for event_index, event in enumerate(episode["events"]):
            close = event["close_frame"]
            group_by_id = {candidate: group_index for group_index, group in enumerate(event["co_motion_groups"]) for candidate in group}
            selected = {c["candidate_id"] for c in event["candidates"] if c["sustained_motion"]}
            cells = []
            for offset in offsets:
                frame = min(max(0, close + offset), len(fixed) - 1)
                left = cv2.cvtColor(fixed[frame], cv2.COLOR_RGB2BGR)
                right = cv2.cvtColor(wrist[frame], cv2.COLOR_RGB2BGR)
                for candidate_id in selected:
                    current = point(tracks[candidate_id][frame])
                    if current is None:
                        continue
                    group = group_by_id.get(candidate_id, 0)
                    color = COLORS[group % len(COLORS)]
                    trail = []
                    for trail_frame in range(max(close, frame - 8), frame + 1):
                        value = point(tracks[candidate_id][trail_frame])
                        if value is not None:
                            trail.append(value)
                    for first, second in zip(trail, trail[1:]):
                        cv2.line(left, first, second, color, 1, cv2.LINE_AA)
                    cv2.circle(left, current, 4, color, -1, cv2.LINE_AA)
                    cv2.putText(left, f"G{group + 1}:{candidate_id}", (current[0] + 4, current[1] - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, .35, color, 1, cv2.LINE_AA)
                pair = np.concatenate([left, right], axis=1)
                title = np.full((27, pair.shape[1], 3), 245, dtype=np.uint8)
                marker = " CLOSE" if offset == 0 else ""
                cv2.putText(title, f"frame {frame} ({offset:+d}){marker}", (5, 18),
                            cv2.FONT_HERSHEY_SIMPLEX, .48, (0, 0, 0), 1, cv2.LINE_AA)
                cells.append(np.concatenate([title, pair], axis=0))
            cell_h, cell_w = cells[0].shape[:2]
            canvas = np.full((3 * cell_h, 4 * cell_w, 3), 255, dtype=np.uint8)
            for index, cell in enumerate(cells):
                row, column = divmod(index, 4)
                canvas[row * cell_h:(row + 1) * cell_h, column * cell_w:(column + 1) * cell_w] = cell
            name = f"{identifier}_close{event_index:02d}_f{close:04d}_motion.jpg"
            cv2.imwrite(str(args.output_dir / name), canvas, [cv2.IMWRITE_JPEG_QUALITY, 90])
            rendered += 1
    print(json.dumps({"rendered_sheets": rendered, "output_dir": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
