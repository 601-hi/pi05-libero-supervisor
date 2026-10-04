"""Export task-blind SAM2 masks for label-blind response windows."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator


def compact(generated: list[dict], shape: tuple[int, int]) -> list[dict]:
    height, width = shape
    selected = []
    for item in sorted(generated, key=lambda x: (-x["predicted_iou"], -x["stability_score"])):
        fraction = float(item["area"] / (height * width))
        x, y, w, h = [float(value) for value in item["bbox"]]
        if not .002 <= fraction <= .08 or y + h / 2 < 72 or y <= 1 or y + h >= height - 1:
            continue
        duplicate = False
        for old in selected:
            ox, oy, ow, oh = old["bbox_xywh"]
            inter = max(0, min(x+w, ox+ow)-max(x, ox)) * max(0, min(y+h, oy+oh)-max(y, oy))
            union = w*h + ow*oh - inter
            if union > 0 and inter / union > .75:
                duplicate = True
                break
        if duplicate:
            continue
        selected.append({
            "bbox_xywh": [x, y, w, h],
            "area_fraction": fraction,
            "predicted_iou": float(item["predicted_iou"]),
            "stability_score": float(item["stability_score"]),
            "mask": np.asarray(item["segmentation"], dtype=bool),
        })
        if len(selected) >= 12:
            break
    return selected


def build_label_blind_jobs(manifest: dict, split: str) -> list[dict]:
    """Build jobs without propagating success, reward, language, or object identity."""
    jobs = []
    for episode in manifest["episodes"]:
        if episode["split"] != split or episode["sidecar_status"] != "matched":
            continue
        if "label_blind_low_response_events" in episode:
            roles = []
            roles.extend(
                (f"event_{index}", window)
                for index, window in enumerate(episode["label_blind_low_response_events"])
            )
            roles.extend(
                (f"control_{index}", window)
                for index, window in enumerate(episode["label_blind_matched_controls"])
            )
        else:
            roles = [
                ("low_response", episode.get("label_blind_low_response_window")),
                ("high_response", episode.get("label_blind_high_response_window")),
            ]
        for role, window in roles:
            if window is not None:
                jobs.append({
                    "episode_id": episode["episode_id"],
                    "sidecar_path": episode["sidecar_path"],
                    "role": role,
                    "action_index": int(window["start_action_index"]),
                })
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", default="development")
    parser.add_argument("--model", default="facebook/sam2.1-hiera-small")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    jobs = build_label_blind_jobs(manifest, args.split)
    generator = SAM2AutomaticMaskGenerator.from_pretrained(
        args.model, points_per_side=24, pred_iou_thresh=.72,
        stability_score_thresh=.82, min_mask_region_area=20,
    )
    records = []
    for job_index, job in enumerate(jobs):
        with np.load(job["sidecar_path"], mmap_mode="r") as archive:
            action_indices = np.asarray(archive["action_indices"])
            matches = np.flatnonzero(action_indices == job["action_index"])
            if len(matches) != 1:
                raise RuntimeError(f"action/frame mapping is not unique: {job}")
            frame_index = int(matches[0])
            image = np.asarray(archive["agent_images"][frame_index], dtype=np.uint8)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            generated = generator.generate(image)
        candidates = compact(generated, image.shape[:2])
        mask_payload = {}
        clean = []
        for candidate_index, candidate in enumerate(candidates):
            mask_payload[f"candidate{candidate_index:02d}"] = candidate.pop("mask")
            clean.append(candidate)
        stem = f"{job['episode_id']}--{job['role']}--a{job['action_index']:03d}"
        np.savez_compressed(args.output_dir / f"{stem}.npz", **mask_payload)
        records.append({
            "episode_id": job["episode_id"], "role": job["role"],
            "action_index": job["action_index"], "frame_index": frame_index,
            "sidecar_path": job["sidecar_path"],
            "mask_path": str((args.output_dir / f"{stem}.npz").resolve()),
            "num_raw_masks": len(generated), "num_candidates": len(clean),
            "candidates": clean,
        })
        if (job_index + 1) % 10 == 0:
            print(f"PROGRESS {job_index + 1}/{len(jobs)}", flush=True)
    result = {
        "schema_version": 1, "model": args.model, "split": args.split,
        "feature_firewall": ["fixed-view RGB", "label-blind action response window"],
        "forbidden_inputs": ["success", "reward", "task language", "target identity"],
        "records": records,
    }
    (args.output_dir / "index.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps({
        "windows": len(records),
        "raw_masks": sum(row["num_raw_masks"] for row in records),
        "candidates": sum(row["num_candidates"] for row in records),
    }))


if __name__ == "__main__":
    main()
