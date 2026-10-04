"""Render compact, annotated contact sheets for recovery failure review."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import cv2
import numpy as np


TASK_RE = re.compile(r"task(\d+)")


def trace_for(task: int, trace_dir: Path) -> Path:
    matches = [p for p in trace_dir.glob(f"*task{task}_*_control.jsonl")
               if not p.name.endswith(".supervisor.jsonl")]
    if len(matches) != 1:
        raise RuntimeError(f"task {task}: expected one trace, got {matches}")
    return matches[0]


def replans(path: Path) -> list[int]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("event") == "step" and (row.get("supervisor_decision") or {}).get("request_replan"):
            out.append(int(row["action_index"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-dir", type=Path, required=True)
    ap.add_argument("--trace-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--tasks", nargs="+", type=int, required=True)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for task in args.tasks:
        videos = list(args.video_dir.glob(f"*task{task:02d}*_*.mp4"))
        if len(videos) != 1:
            raise RuntimeError(f"task {task}: expected one video, got {videos}")
        cap = cv2.VideoCapture(str(videos[0]))
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        key = set(np.linspace(0, max(0, count - 1), 12).round().astype(int).tolist())
        rp = replans(trace_for(task, args.trace_dir))
        for value in rp:
            key.update(i for i in (value - 1, value, value + 1, value + 10) if 0 <= i < count)
        indices = sorted(key)
        cells = []
        for index in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = cap.read()
            if not ok:
                continue
            frame = cv2.resize(frame, (256, 256), interpolation=cv2.INTER_AREA)
            label = f"task {task} step {index}"
            if index in rp:
                label += " REPLAN"
            cv2.rectangle(frame, (0, 0), (256, 25), (0, 0, 0), -1)
            cv2.putText(frame, label, (5, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.43,
                        (0, 255, 255) if index in rp else (255, 255, 255), 1, cv2.LINE_AA)
            cells.append(frame)
        cap.release()
        cols = 4
        rows = (len(cells) + cols - 1) // cols
        blank = np.zeros_like(cells[0])
        cells += [blank] * (rows * cols - len(cells))
        sheet = np.vstack([np.hstack(cells[r * cols:(r + 1) * cols]) for r in range(rows)])
        cv2.imwrite(str(args.output_dir / f"task{task:02d}_failure_contact_sheet.jpg"), sheet,
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(task, count, rp, len(indices))


if __name__ == "__main__":
    main()
