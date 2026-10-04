#!/usr/bin/env python3
"""Export causal third-person visual motion features aligned to dynamics NPZ rows."""
from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

import cv2
import numpy as np


def grid_means(array: np.ndarray, size: int = 4) -> np.ndarray:
    h, w = array.shape[:2]
    rows = []
    for iy in range(size):
        for ix in range(size):
            cell = array[iy*h//size:(iy+1)*h//size, ix*w//size:(ix+1)*w//size]
            rows.append(cell.reshape(-1, *array.shape[2:]).mean(0) if array.ndim == 3 else cell.mean())
    return np.asarray(rows).reshape(-1)


def frame_feature(previous: np.ndarray, current: np.ndarray) -> np.ndarray:
    prev = cv2.resize(previous, (112, 112), interpolation=cv2.INTER_AREA)
    curr = cv2.resize(current, (112, 112), interpolation=cv2.INTER_AREA)
    pg = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    cg = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(pg, cg, None, .5, 3, 15, 3, 5, 1.2, 0)
    magnitude = np.linalg.norm(flow, axis=2)
    difference = cv2.absdiff(pg, cg).astype(np.float32) / 255.0
    context_gray = cv2.resize(cg, (8, 8), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    context_color = cv2.resize(curr, (4, 4), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    return np.r_[grid_means(flow[..., 0]), grid_means(flow[..., 1]),
                 grid_means(magnitude), grid_means(difference),
                 context_gray.ravel(), context_color.ravel()]


def read_video(path: str) -> list[np.ndarray]:
    capture = cv2.VideoCapture(path); frames = []
    while True:
        ok, frame = capture.read()
        if not ok: break
        frames.append(frame)
    capture.release()
    return frames


def scale_tag(value: float) -> str:
    return f"{value:.3f}".replace(".", "p")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--video-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    d = np.load(a.dataset, allow_pickle=False)
    meta = json.loads(str(d["metadata_json"])); sources = meta["sources"]
    visual = np.zeros((len(d["x"]), 176), dtype=np.float32)
    valid = np.zeros(len(d["x"]), dtype=bool); paths = {}
    for source in sources:
        sid, trace = int(source["source_id"]), Path(source["path"])
        seed_match = re.search(r"seed(\d+)", trace.name)
        if not seed_match: raise ValueError(f"cannot infer seed from {trace}")
        seed = int(seed_match.group(1))
        rows = [json.loads(line) for line in trace.open(encoding="utf-8") if line.strip()]
        for task in range(10):
            for episode in sorted({int(r["episode_idx"]) for r in rows if r.get("event") == "step" and int(r["task_id"]) == task}):
                steps = [r for r in rows if r.get("event") == "step" and int(r["task_id"]) == task and int(r["episode_idx"]) == episode]
                steps.sort(key=lambda r: int(r["action_index"]))
                eid = f"source{sid}:task{task}:episode{episode}"
                idx = np.flatnonzero(d["episode_id"] == eid)
                idx = idx[np.argsort(d["action_index"][idx])]
                if len(idx) != len(steps): raise ValueError(f"row mismatch {eid}: {len(idx)} vs {len(steps)}")
                planned_scale = float(d["scale"][idx[0]])
                start = int(steps[0]["scheduled_disturbance_start"])
                nsteps = 0 if np.isclose(planned_scale, 1.0) else 10
                pattern = str(a.video_dir / (f"rollout_seed{seed}_noise*_task{task:02d}_episode{episode:03d}_"
                    f"random_start{start:03d}_n{nsteps:03d}_scale{scale_tag(planned_scale)}_*.mp4"))
                matches = glob.glob(pattern)
                if len(matches) != 1: raise ValueError(f"expected one video for {eid}, found {matches}, pattern={pattern}")
                frames = read_video(matches[0])
                if len(frames) != len(steps): raise ValueError(f"frame mismatch {eid}: {len(frames)} vs {len(steps)}")
                # Images are captured immediately before env.step(action_j).
                # Therefore the visual consequence of action_j is frame_j ->
                # frame_{j+1}; the final action has no post-action image.
                for j, i in enumerate(idx):
                    if j + 1 < len(frames):
                        visual[i] = frame_feature(frames[j], frames[j + 1])
                        valid[i] = True
                paths[eid] = matches[0]
    if not valid.all() and int((~valid).sum()) != len(np.unique(d["episode_id"])):
        raise AssertionError("exactly the final action of each episode should be invalid")
    np.savez_compressed(a.out, visual=visual, valid=valid, x_condition=d["x"],
                        abnormal=d["abnormal"], task_id=d["task_id"], episode_id=d["episode_id"],
                        action_index=d["action_index"], scale=d["scale"], metadata_json=np.asarray(json.dumps({
                            "feature_dim": 176, "causal": True, "camera": "agentview_mp4",
                            "alignment": "action_j -> pre_action_frame_j to pre_action_frame_j_plus_1",
                            "feature_groups": {"flow_grid_xy_magnitude_and_diff": 64, "gray_context_8x8": 64, "color_context_4x4": 48},
                            "videos": paths}, ensure_ascii=False)))
    print(json.dumps({"rows": len(visual), "valid_rows": int(valid.sum()), "episodes": len(paths),
                      "feature_dim": visual.shape[1]}, indent=2))


if __name__ == "__main__": main()
