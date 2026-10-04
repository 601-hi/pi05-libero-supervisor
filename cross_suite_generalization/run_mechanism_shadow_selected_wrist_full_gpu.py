#!/usr/bin/env python3
"""Track only frozen high-quality wrist candidates from close to episode end."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cross_suite_generalization.run_wrist_close_candidates_gpu import MODEL, track  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--public", type=Path, required=True)
    parser.add_argument("--private", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    public = json.loads(args.public.read_text(encoding="utf-8"))["records"]
    private = {
        row["anonymous_id"]: row
        for row in json.loads(args.private.read_text(encoding="utf-8"))["records"]
    }
    scores = {
        row["anonymous_id"]: row
        for row in json.loads(args.scores.read_text(encoding="utf-8"))["records"]
    }
    identifiers = {row["anonymous_id"] for row in public}
    if identifiers != set(private) or identifiers != set(scores):
        raise ValueError("public/private/scores anonymous ids do not match")
    selected = [value for value in identifiers if scores[value].get("passes_frozen_candidate_gate")]
    for identifier in identifiers:
        if not Path(private[identifier]["original_sidecar"]).is_file():
            raise FileNotFoundError(private[identifier]["original_sidecar"])
    for identifier in selected:
        if not (args.candidate_dir / f"{identifier}_initial_masks.npz").is_file():
            raise FileNotFoundError(args.candidate_dir / f"{identifier}_initial_masks.npz")
    if args.dry_run:
        print(json.dumps({"episodes": len(public), "selected": len(selected),
                          "abstained": len(public) - len(selected)}))
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    from sam2.sam2_video_predictor import SAM2VideoPredictor

    args.output_dir.mkdir(parents=True, exist_ok=True)
    final_path = args.output_dir / "predictions.json"
    partial_path = args.output_dir / "predictions.partial.json"
    if final_path.exists():
        raise FileExistsError(final_path)
    records = (
        json.loads(partial_path.read_text(encoding="utf-8"))["records"]
        if args.resume and partial_path.exists() else []
    )
    completed = {row["anonymous_id"] for row in records}
    predictor = SAM2VideoPredictor.from_pretrained(MODEL)
    for episode_number, row in enumerate(public, 1):
        identifier = row["anonymous_id"]
        if identifier in completed:
            continue
        score = scores[identifier]
        if not score.get("passes_frozen_candidate_gate"):
            records.append({
                "anonymous_id": identifier,
                "status": "candidate_gate_abstained",
                "selection_state": score["state"],
            })
        else:
            candidate_set = score["candidate_set"]
            if len(candidate_set) != 1:
                raise ValueError(f"{identifier}: gate passed without unique candidate")
            candidate_id = int(candidate_set[0])
            close = int(score["close_frame"])
            with np.load(private[identifier]["original_sidecar"], allow_pickle=False) as archive:
                frames = np.asarray(archive["wrist_images"][close:], dtype=np.uint8)
            with np.load(
                args.candidate_dir / f"{identifier}_initial_masks.npz", allow_pickle=False
            ) as archive:
                masks = np.asarray(archive["masks"], dtype=np.uint8)
                boxes = np.asarray(archive["boxes_xywh"], dtype=float)
            if candidate_id < 1 or candidate_id > len(boxes):
                raise ValueError(f"{identifier}: selected candidate is missing")
            selected = [{"box": boxes[candidate_id - 1].tolist(), "mask": masks[candidate_id - 1]}]
            tracks, areas = track(predictor, frames, selected)
            records.append({
                "anonymous_id": identifier,
                "status": "tracked",
                "close_frame": close,
                "tracked_frames": len(frames),
                "source_wrist_candidate_id": candidate_id,
                "candidate_quality_probability": score["controlled_object_candidate_probability"],
                "centroids_xy": tracks[1],
                "area_fraction": areas[1],
            })
        partial_path.write_text(
            json.dumps({"schema_version": 1, "records": records}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({"episode": episode_number, "anonymous_id": identifier,
                          "status": records[-1]["status"]}), flush=True)
    result = {
        "schema_version": 1,
        "protocol": {
            "model": MODEL,
            "candidate_source": "frozen unique controlled-object candidate above gate",
            "anchor": "first close",
            "stop": "episode end",
            "outcome_labels_used": False,
        },
        "records": records,
    }
    final_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
