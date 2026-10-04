#!/usr/bin/env python3
"""Track the predicted controlled wrist object from first close to episode end."""
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
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--bridge", type=Path, required=True)
    parser.add_argument("--wrist-candidates", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    from sam2.sam2_video_predictor import SAM2VideoPredictor

    public = json.loads(args.public.read_text(encoding="utf-8"))["records"]
    private = {row["anonymous_id"]: row for row in json.loads(args.private.read_text(encoding="utf-8"))["records"]}
    features = {row["anonymous_id"]: row for row in json.loads(args.features.read_text(encoding="utf-8"))["records"]}
    bridge = {row["anonymous_id"]: row for row in json.loads(args.bridge.read_text(encoding="utf-8"))["records"]}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    final_path = args.output_dir / "predictions.json"
    if final_path.exists():
        raise FileExistsError(final_path)
    predictor = SAM2VideoPredictor.from_pretrained(MODEL)
    records = []
    for episode_number, row in enumerate(public, 1):
        identifier = row["anonymous_id"]
        bridge_row = bridge[identifier]["physical_ranked_pipeline_bridge"]
        closes = features[identifier]["close_frames"]
        chosen_id = None if bridge_row is None else int(bridge_row["chosen_wrist_id"])
        if not closes or chosen_id is None:
            records.append({"anonymous_id": identifier, "status": "no_close_or_bridge"})
            continue
        close = int(closes[0])
        with np.load(private[identifier]["original_sidecar"], allow_pickle=False) as data:
            wrist = np.asarray(data["wrist_images"], dtype=np.uint8)
        with np.load(args.wrist_candidates / f"{identifier}_initial_masks.npz", allow_pickle=False) as data:
            masks = np.asarray(data["masks"], dtype=np.uint8)
            boxes = np.asarray(data["boxes_xywh"], dtype=float)
        if chosen_id < 1 or chosen_id > len(boxes):
            records.append({"anonymous_id": identifier, "status": "chosen_candidate_missing",
                            "chosen_wrist_id": chosen_id})
            continue
        selected = [{"box": boxes[chosen_id - 1].tolist(), "mask": masks[chosen_id - 1]}]
        frames = wrist[close:]
        tracks, areas = track(predictor, frames, selected)
        records.append({
            "anonymous_id": identifier,
            "status": "tracked",
            "close_frame": close,
            "episode_frames": int(len(wrist)),
            "tracked_frames": int(len(frames)),
            "source_wrist_candidate_id": chosen_id,
            "centroids_xy": tracks[1],
            "area_fraction": areas[1],
        })
        partial = {"schema_version": 1, "records": records}
        (args.output_dir / "predictions.partial.json").write_text(
            json.dumps(partial, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps({"episode": episode_number, "anonymous_id": identifier,
                          "frames": len(frames), "source_candidate": chosen_id}), flush=True)
    result = {
        "schema_version": 1,
        "protocol": {
            "model": MODEL,
            "candidate_source": "frozen physical-ranked cross-view bridge",
            "anchor": "first close frame",
            "stop": "episode end",
            "outcome_labels_used": False,
        },
        "records": records,
    }
    final_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
