#!/usr/bin/env python3
"""Render frozen holdout false-trigger windows for human diagnosis."""
from pathlib import Path
import json

import cv2
import numpy as np


TRACE_ROOT = Path("/root/gpufree-data/libero-traces/natural_validation_seed35_v1/visual_sidecars")
OUTPUT_ROOT = Path("/root/gpufree-data/supervisor-tools-v1/outputs/task0_close_relation_20260927")


for episode, start, end in ((0, 40, 100), (2, 96, 172)):
    sidecar = TRACE_ROOT / (
        f"rollout_libero_90_seed35_noise2026091835_task00_episode00{episode}_"
        "fixed_startnone_n000_scale1p000_failure.npz"
    )
    scores = json.loads(
        (OUTPUT_ROOT / f"holdout_failure35_ep{episode}_counterstate.json").read_text()
    )
    by_frame = {row["frame"]: row for row in scores["records"]}
    with np.load(sidecar, mmap_mode="r") as data:
        images = []
        for frame in range(start, end + 1, 8):
            image = cv2.cvtColor(np.asarray(data["agent_images"][frame]), cv2.COLOR_RGB2BGR)
            image = cv2.resize(image, (448, 448), interpolation=cv2.INTER_NEAREST)
            row = by_frame.get(frame)
            label = str(frame) if row is None else f'{frame} iou={row["selected_iou_to_anchor"]:.2f}'
            cv2.putText(image, label, (8, 30), cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 255, 255), 2)
            if row and row["selected_box_xyxy"]:
                box = (np.asarray(row["selected_box_xyxy"]) * 2).astype(int)
                cv2.rectangle(image, tuple(box[:2]), tuple(box[2:]), (0, 0, 255), 2)
            images.append(image)
    rows = []
    for offset in range(0, len(images), 4):
        row = images[offset:offset + 4]
        while len(row) < 4:
            row.append(np.zeros_like(images[0]))
        rows.append(np.concatenate(row, axis=1))
    cv2.imwrite(str(OUTPUT_ROOT / f"holdout_failure_ep{episode}_false_trigger.jpg"), np.concatenate(rows))
