"""Render anonymous fixed-view videos for physical-motion ground-truth review."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


COLORS = [
    (220, 200, 0), (20, 190, 245), (160, 60, 230), (80, 200, 60),
    (20, 130, 240), (230, 60, 130), (220, 160, 20), (60, 80, 230),
    (180, 180, 40), (40, 180, 180), (180, 40, 180), (80, 140, 240),
]


def valid_point(value: object) -> tuple[int, int] | None:
    if not isinstance(value, list) or len(value) != 2 or value[0] is None or value[1] is None:
        return None
    xy = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(xy)):
        return None
    return int(round(xy[0])), int(round(xy[1]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--private-map", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--trail", type=int, default=12)
    args = parser.parse_args()

    private = {r["anonymous_id"]: r for r in json.loads(args.private_map.read_text(encoding="utf-8"))["records"]}
    predictions = json.loads(args.predictions.read_text(encoding="utf-8"))["records"]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for record in predictions:
        identifier = record["anonymous_id"]
        with np.load(private[identifier]["original_sidecar"], allow_pickle=False) as data:
            frames = np.asarray(data["agent_images"], dtype=np.uint8)
        height, width = frames.shape[1:3]
        output = args.output_dir / f"{identifier}_motion_review.mp4"
        writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (2 * width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Could not open video writer for {output}")
        histories: dict[str, list[tuple[int, int] | None]] = {}
        for candidate_id, sequence in record["centroids_xy"].items():
            histories[candidate_id] = [valid_point(value) for value in sequence]

        for frame_index, rgb in enumerate(frames):
            left = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            right = left.copy()
            for candidate_id, points in histories.items():
                color = COLORS[(int(candidate_id) - 1) % len(COLORS)]
                start = max(0, frame_index - args.trail + 1)
                trail = [point for point in points[start : frame_index + 1] if point is not None]
                for first, second in zip(trail, trail[1:]):
                    cv2.line(right, first, second, color, 1, cv2.LINE_AA)
                current = points[frame_index] if frame_index < len(points) else None
                if current is not None:
                    cv2.circle(right, current, 3, color, -1, cv2.LINE_AA)
                    cv2.putText(right, candidate_id, (current[0] + 4, current[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)
            cv2.putText(left, f"frame {frame_index}", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(right, "candidate tracks", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
            writer.write(np.concatenate([left, right], axis=1))
        writer.release()
        print(json.dumps({"anonymous_id": identifier, "frames": len(frames), "output": str(output)}))


if __name__ == "__main__":
    main()
