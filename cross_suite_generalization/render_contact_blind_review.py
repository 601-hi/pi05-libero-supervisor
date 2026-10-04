"""Render anonymous dual-view videos for Pass-A physical contact annotation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def render_episode(sidecar: Path, output: Path, close_frames: list[int], fps: float) -> dict:
    data = np.load(sidecar, allow_pickle=False)
    fixed = data["agent_images"]
    wrist = data["wrist_images"]
    actions = data["action_indices"]
    if len(fixed) != len(wrist) or len(fixed) != len(actions):
        raise ValueError(f"unaligned sidecar arrays: {sidecar}")
    if fixed.shape[1:] != wrist.shape[1:]:
        raise ValueError(f"camera shapes disagree: {sidecar}")
    height, width = fixed.shape[1:3]
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width * 2, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"failed to open video writer: {output}")
    close_set = set(close_frames)
    try:
        for visual_frame, (left, right, action_index) in enumerate(zip(fixed, wrist, actions)):
            canvas = np.concatenate([left, right], axis=1)
            # Stored images are RGB; OpenCV video expects BGR.
            canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
            cv2.putText(canvas, "fixed/world", (6, 17), cv2.FONT_HERSHEY_SIMPLEX,
                        .45, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(canvas, "wrist/gripper", (width + 6, 17), cv2.FONT_HERSHEY_SIMPLEX,
                        .45, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(canvas, f"frame {visual_frame:04d} action {int(action_index):04d}",
                        (6, height - 8), cv2.FONT_HERSHEY_SIMPLEX, .42,
                        (255, 255, 255), 1, cv2.LINE_AA)
            if visual_frame in close_set:
                cv2.rectangle(canvas, (0, 0), (width * 2 - 1, height - 1), (0, 215, 255), 3)
                cv2.putText(canvas, "CLOSE EVENT", (width - 55, 18),
                            cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 215, 255), 1, cv2.LINE_AA)
            writer.write(canvas)
    finally:
        writer.release()
    return {
        "frames": len(fixed),
        "first_action_index": int(actions[0]) if len(actions) else None,
        "last_action_index": int(actions[-1]) if len(actions) else None,
        "close_frames": close_frames,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--private-map", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--index-output", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=10.0)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))["records"]
    private = {
        row["anonymous_id"]: row
        for row in json.loads(args.private_map.read_text(encoding="utf-8"))["records"]
    }
    rows = []
    for item in manifest:
        identifier = item["anonymous_id"]
        if identifier not in private:
            raise KeyError(f"missing private sidecar mapping: {identifier}")
        output = args.output_dir / f"{identifier}.mp4"
        metadata = render_episode(
            Path(private[identifier]["original_sidecar"]),
            output,
            [int(value) for value in item.get("close_frames", [])],
            args.fps,
        )
        rows.append({
            "anonymous_id": identifier,
            "development_wave": item["development_wave"],
            "video": output.name,
            **metadata,
        })
    index = {
        "schema_version": 1,
        "information_visible": ["anonymous id", "fixed RGB", "wrist RGB", "frame", "action index", "close-event marker"],
        "information_hidden": ["task language", "outcome", "reward", "expert score", "model-selected candidate"],
        "records": rows,
    }
    args.index_output.parent.mkdir(parents=True, exist_ok=True)
    args.index_output.write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"videos": len(rows), "frames": sum(row["frames"] for row in rows)}, indent=2))


if __name__ == "__main__":
    main()
