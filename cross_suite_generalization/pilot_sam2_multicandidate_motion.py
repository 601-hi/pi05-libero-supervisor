"""Track compact first-frame masks and rank candidates by visual motion."""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sam2.sam2_video_predictor import SAM2VideoPredictor


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--goal-indices", default="2,3,5,6,7")
    parser.add_argument("--max-candidates", type=int, default=12)
    parser.add_argument("--model", default="facebook/sam2.1-hiera-small")
    return parser.parse_args()


def box_iou(left: list[float], right: list[float]) -> float:
    lx, ly, lw, lh = left
    rx, ry, rw, rh = right
    intersection = max(0.0, min(lx + lw, rx + rw) - max(lx, rx)) * max(
        0.0, min(ly + lh, ry + rh) - max(ly, ry)
    )
    return intersection / max(lw * lh + rw * rh - intersection, 1e-9)


def select_boxes(candidates: list[dict], max_candidates: int) -> list[list[float]]:
    selected = []
    for item in candidates:
        box = item["bbox_xywh"]
        x, y, width, height = box
        center_y = y + height / 2
        # Side-border objects are legitimate in LIBERO (e.g. a bowl on the
        # cabinet at image right). Reject only top/bottom truncation here.
        if center_y < 72 or y <= 1 or y + height >= 223:
            continue
        if any(box_iou(box, previous) > 0.75 for previous in selected):
            continue
        selected.append(box)
        if len(selected) >= max_candidates:
            break
    return selected


def unique_goal_jobs(jobs: list[dict]) -> dict[str, dict]:
    result = {}
    for job in jobs:
        result.setdefault(str(job["goal_language"]), {"sidecar": job["sidecar"]})
    return result


def centroid(mask: np.ndarray) -> np.ndarray:
    points = np.argwhere(mask)
    return points.mean(axis=0)[::-1] if len(points) else np.array([np.nan, np.nan])


def main() -> None:
    args = arguments()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    allowed_goal_indices = {int(value) for value in args.goal_indices.split(",") if value}
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    candidate_data = json.loads(args.candidates.read_text(encoding="utf-8"))
    jobs = unique_goal_jobs(manifest["jobs"])
    predictor = SAM2VideoPredictor.from_pretrained(args.model)
    episode_results = []

    for record in candidate_data["records"]:
        goal_index = int(record["goal_index"])
        if goal_index not in allowed_goal_indices:
            continue
        boxes_xywh = select_boxes(record["candidates"], args.max_candidates)
        goal = record["goal_language"]
        with np.load(jobs[goal]["sidecar"], mmap_mode="r") as sidecar:
            frames = np.asarray(sidecar["agent_images"], dtype=np.uint8)
        with tempfile.TemporaryDirectory(prefix="sam2_multi_") as temporary:
            frame_dir = Path(temporary)
            for frame_index, frame in enumerate(frames):
                Image.fromarray(frame).save(frame_dir / f"{frame_index:05d}.jpg", quality=95)
            state = predictor.init_state(video_path=str(frame_dir), async_loading_frames=False)
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                for object_id, (x, y, width, height) in enumerate(boxes_xywh, start=1):
                    predictor.add_new_points_or_box(
                        inference_state=state,
                        frame_idx=0,
                        obj_id=object_id,
                        box=np.asarray([x, y, x + width, y + height], dtype=np.float32),
                    )
                masks_by_frame = {}
                for frame_index, object_ids, mask_logits in predictor.propagate_in_video(state):
                    masks_by_frame[int(frame_index)] = {
                        int(object_id): (mask_logits[position, 0].detach().cpu().numpy() > 0)
                        for position, object_id in enumerate(object_ids)
                    }
            predictor.reset_state(state)

        metrics = []
        diagonal = float(np.hypot(*frames.shape[1:3]))
        for object_id, box in enumerate(boxes_xywh, start=1):
            masks = [
                masks_by_frame.get(frame_index, {}).get(
                    object_id, np.zeros(frames.shape[1:3], dtype=bool)
                )
                for frame_index in range(len(frames))
            ]
            centers = np.asarray([centroid(mask) for mask in masks])
            valid = np.all(np.isfinite(centers), axis=1)
            valid_centers = centers[valid]
            if len(valid_centers):
                displacement = np.linalg.norm(valid_centers - valid_centers[0], axis=1)
                max_displacement = float(displacement.max() / diagonal)
                path = float(np.linalg.norm(np.diff(valid_centers, axis=0), axis=1).sum() / diagonal)
            else:
                max_displacement = path = 0.0
            areas = np.asarray([mask.mean() for mask in masks])
            metrics.append({
                "object_id": object_id,
                "initial_box_xywh": box,
                "max_centroid_displacement_normalized": max_displacement,
                "centroid_path_normalized": path,
                "visible_fraction": float((areas > 0.001).mean()),
                "median_area_fraction": float(np.median(areas)),
                "centroid_xy": [
                    [float(point[0]), float(point[1])] if np.all(np.isfinite(point)) else None
                    for point in centers
                ],
            })
        metrics.sort(key=lambda item: item["max_centroid_displacement_normalized"], reverse=True)
        top_ids = [item["object_id"] for item in metrics[:3]]
        review_indices = sorted(set((0, len(frames) // 4, len(frames) // 2, 3 * len(frames) // 4, len(frames) - 1)))
        colors = {top_ids[index]: np.asarray(color, dtype=float) for index, color in enumerate(
            ((255, 40, 40), (40, 255, 40), (40, 160, 255))[: len(top_ids)]
        )}
        for frame_index in review_indices:
            canvas = frames[frame_index].astype(float).copy()
            for object_id, color in colors.items():
                mask = masks_by_frame.get(frame_index, {}).get(
                    object_id, np.zeros(frames.shape[1:3], dtype=bool)
                )
                canvas[mask] = 0.45 * canvas[mask] + 0.55 * color
            Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8)).save(
                args.output_dir / f"goal{goal_index:02d}_frame{frame_index:04d}.png"
            )
        episode_results.append({
            "goal_index": goal_index,
            "goal_language": goal,
            "num_frames": len(frames),
            "num_tracked_candidates": len(boxes_xywh),
            "review_indices": review_indices,
            "motion_ranked_candidates": metrics,
        })

    result = {
        "schema_version": 1,
        "model": args.model,
        "feature_firewall": ["agent RGB sequence"],
        "selection": "compact first-frame automatic masks ranked by visual centroid motion",
        "episodes": episode_results,
    }
    (args.output_dir / "motion_ranking.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps({
        "episodes": len(episode_results),
        "frames": sum(row["num_frames"] for row in episode_results),
        "tracked_candidates": sum(row["num_tracked_candidates"] for row in episode_results),
    }))


if __name__ == "__main__":
    main()
