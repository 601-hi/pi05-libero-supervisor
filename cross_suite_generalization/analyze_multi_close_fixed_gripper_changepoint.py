"""Measure whether a fixed-view candidate becomes gripper-coupled after closure.

Robot pixels that already co-move with the EEF before closure should have little
change.  A newly controlled object should show a drop in candidate-minus-EFF
relative speed and usually a rise in global image speed after closure.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from cross_suite_generalization.analyze_fixed_gripper_changepoint import log_ratio, median_speed, points
from vla_supervisor.camera_geometry import CameraCalibration, project_world_point


def project_track(sidecar: str, indices: list[int]) -> np.ndarray:
    result = []
    with np.load(sidecar, allow_pickle=False, mmap_mode="r") as data:
        height, width = map(int, data["saved_image_shape"])
        for index in indices:
            intrinsic_values = data["agent_camera_intrinsic"]
            intrinsic = np.asarray(
                intrinsic_values if intrinsic_values.shape == (3, 3) else intrinsic_values[index],
                dtype=float,
            )
            camera_to_world = np.asarray(data["agent_camera_to_world"][index], dtype=float)
            calibration = CameraCalibration(
                calibration_id="sidecar-frame", camera_id="agentview",
                intrinsic=tuple(map(tuple, intrinsic)),
                world_to_camera=tuple(map(tuple, np.linalg.inv(camera_to_world))),
                image_height=height, image_width=width,
                flip_x=bool(np.asarray(data["projection_flip_x"]).item()),
                flip_y=bool(np.asarray(data["projection_flip_y"]).item()),
            )
            projection = project_world_point(data["eef_positions"][index], calibration)
            result.append(projection.xy if projection.visible else (np.nan, np.nan))
    return np.asarray(result, dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--post-settle", type=int, default=5)
    parser.add_argument("--post-window", type=int, default=25)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    sidecars = {row["anonymous_id"]: row["sidecar"] for row in manifest["episodes"]}
    records = json.loads(args.predictions.read_text(encoding="utf-8"))["records"]
    outputs = []
    for record in records:
        pre_indices = list(range(int(record["window_start"]), int(record["close_frame"]) + 1))
        post_indices = list(range(int(record["close_frame"]), int(record["window_end"]) + 1))
        pre_gripper = project_track(sidecars[record["anonymous_id"]], pre_indices)
        post_gripper = project_track(sidecars[record["anonymous_id"]], post_indices)
        candidates = []
        for candidate_id in record["post_centroids_xy"]:
            pre = points(record["pre_centroids_xy"][candidate_id])
            post = points(record["post_centroids_xy"][candidate_id])
            pre_relative, post_relative = pre - pre_gripper[:len(pre)], post - post_gripper[:len(post)]
            pre_rel = median_speed(pre_relative, 0, len(pre_relative))
            post_rel = median_speed(post_relative, args.post_settle, args.post_settle + args.post_window)
            pre_global = median_speed(pre, 0, len(pre))
            post_global = median_speed(post, args.post_settle, args.post_settle + args.post_window)
            relative_drop = log_ratio(pre_rel, post_rel)
            global_rise = -log_ratio(pre_global, post_global)
            candidates.append({
                "candidate_id": candidate_id,
                "pre_relative_speed_px": pre_rel, "post_relative_speed_px": post_rel,
                "relative_speed_log_drop": relative_drop,
                "pre_global_speed_px": pre_global, "post_global_speed_px": post_global,
                "global_speed_log_rise": global_rise,
                "combined_change": relative_drop + global_rise,
            })
        candidates.sort(key=lambda row: np.nan_to_num(row["combined_change"], nan=-1e9), reverse=True)
        outputs.append({
            "event_id": record["event_id"], "anonymous_id": record["anonymous_id"],
            "close_ordinal": record["close_ordinal"], "close_frame": record["close_frame"],
            "candidates": candidates,
        })
    result = {
        "schema_version": 1,
        "interpretation": "descriptive fixed-view candidate/EFF change point; uncalibrated on these events",
        "warning": "combined_change is not a probability and has no deployment threshold here",
        "parameters": {"post_settle": args.post_settle, "post_window": args.post_window},
        "records": outputs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    print(json.dumps({"events": len(outputs), "candidates": sum(len(r["candidates"]) for r in outputs)}))


if __name__ == "__main__":
    main()
