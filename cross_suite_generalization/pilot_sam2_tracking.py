"""Track only non-ambiguous Florence-seeded objects with SAM 2."""
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
    parser.add_argument("--grounding", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="facebook/sam2.1-hiera-small")
    return parser.parse_args()


def unique_goal_jobs(jobs: list[dict]) -> dict[str, dict]:
    selected = {}
    for job in jobs:
        goal = str(job["goal_language"])
        selected.setdefault(goal, {"sidecar": job["sidecar"], "goal_language": goal})
    return selected


def overlay(image: np.ndarray, mask: np.ndarray) -> Image.Image:
    result = image.astype(np.float32).copy()
    color = np.array([255.0, 40.0, 40.0], dtype=np.float32)
    result[mask] = 0.45 * result[mask] + 0.55 * color
    return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))


def main() -> None:
    args = arguments()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    grounding = json.loads(args.grounding.read_text(encoding="utf-8"))
    jobs = unique_goal_jobs(manifest["jobs"])
    predictor = SAM2VideoPredictor.from_pretrained(args.model)
    records = []

    for grounded in grounding["records"]:
        selection = grounded["source_consistent_selection"]
        selected_index = selection["object_prediction_index"]
        if selection["ambiguous"] or selected_index is None:
            continue
        goal = grounded["goal_language"]
        job = jobs[goal]
        initial_box = grounded["predictions"][selected_index]["box_xyxy"]
        with np.load(job["sidecar"], mmap_mode="r") as sidecar:
            frames = np.asarray(sidecar["agent_images"], dtype=np.uint8)
        with tempfile.TemporaryDirectory(prefix="sam2_frames_") as temporary:
            frame_dir = Path(temporary)
            for frame_index, frame in enumerate(frames):
                Image.fromarray(frame).save(frame_dir / f"{frame_index:05d}.jpg", quality=95)
            state = predictor.init_state(video_path=str(frame_dir), async_loading_frames=False)
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                predictor.add_new_points_or_box(
                    inference_state=state,
                    frame_idx=0,
                    obj_id=1,
                    box=np.asarray(initial_box, dtype=np.float32),
                )
                masks = {}
                for frame_index, object_ids, mask_logits in predictor.propagate_in_video(state):
                    object_position = list(object_ids).index(1)
                    masks[int(frame_index)] = (
                        mask_logits[object_position, 0].detach().cpu().numpy() > 0
                    )
            predictor.reset_state(state)

        review_indices = sorted(set((0, len(frames) // 4, len(frames) // 2, 3 * len(frames) // 4, len(frames) - 1)))
        areas = []
        for frame_index in range(len(frames)):
            mask = masks.get(frame_index, np.zeros(frames.shape[1:3], dtype=bool))
            areas.append(float(mask.mean()))
            if frame_index in review_indices:
                overlay(frames[frame_index], mask).save(
                    args.output_dir / f'goal{grounded["goal_index"]:02d}_frame{frame_index:04d}.png'
                )
        records.append({
            "goal_index": grounded["goal_index"],
            "goal_language": goal,
            "initial_box_xyxy": initial_box,
            "num_frames": len(frames),
            "review_indices": review_indices,
            "mask_area_fraction": areas,
        })

    result = {
        "schema_version": 1,
        "model": args.model,
        "seed_policy": "only non-ambiguous source-consistent Florence selections",
        "feature_firewall": ["agent RGB sequence", "goal_language-derived initial box"],
        "records": records,
    }
    (args.output_dir / "tracking.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps({"model": args.model, "episodes": len(records), "frames": sum(r["num_frames"] for r in records)}))


if __name__ == "__main__":
    main()
