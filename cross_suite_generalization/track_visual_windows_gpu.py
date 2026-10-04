"""Track frozen development candidates for five frames and measure visual consequence."""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sam2.sam2_video_predictor import SAM2VideoPredictor

from vla_supervisor.background_motion import estimate_background_motion, residual_motion_in_mask


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window-frames", type=int, default=5)
    parser.add_argument("--model", default="facebook/sam2.1-hiera-small")
    args = parser.parse_args()
    index = json.loads(args.candidate_index.read_text(encoding="utf-8"))
    predictor = SAM2VideoPredictor.from_pretrained(args.model)
    results = []
    for record_index, record in enumerate(index["records"]):
        with np.load(record["sidecar_path"], mmap_mode="r") as archive:
            start = int(record["frame_index"])
            stop = min(start + args.window_frames, len(archive["agent_images"]))
            frames = np.asarray(archive["agent_images"][start:stop], dtype=np.uint8)
        with np.load(record["mask_path"]) as masks:
            seed_masks = [np.asarray(masks[key], dtype=bool) for key in sorted(masks.files)]
        if len(frames) < 2 or not seed_masks:
            results.append({
                "episode_id": record["episode_id"], "role": record["role"],
                "action_index": record["action_index"], "status": "insufficient_frames_or_candidates",
                "steps": [],
            })
            continue
        with tempfile.TemporaryDirectory(prefix="visual_window_") as temporary:
            frame_dir = Path(temporary)
            for frame_index, frame in enumerate(frames):
                Image.fromarray(frame).save(frame_dir / f"{frame_index:05d}.jpg", quality=95)
            state = predictor.init_state(video_path=str(frame_dir), async_loading_frames=False)
            tracked = {}
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                for object_id, mask in enumerate(seed_masks, start=1):
                    predictor.add_new_mask(state, 0, object_id, mask)
                for frame_index, object_ids, logits in predictor.propagate_in_video(
                    state, start_frame_idx=0, max_frame_num_to_track=len(frames)
                ):
                    tracked[int(frame_index)] = {
                        int(object_id): (logits[position, 0].detach().cpu().numpy() > 0)
                        for position, object_id in enumerate(object_ids)
                    }
            predictor.reset_state(state)
        steps = []
        for relative in range(len(frames) - 1):
            estimate = estimate_background_motion(frames[relative], frames[relative + 1])
            candidate_measurements = []
            for object_id in range(1, len(seed_masks) + 1):
                mask = tracked.get(relative, {}).get(object_id)
                if mask is None or not estimate.valid:
                    continue
                residual = residual_motion_in_mask(frames[relative], frames[relative + 1], mask, estimate)
                if residual.valid_pixels:
                    candidate_measurements.append({
                        "object_id": object_id,
                        "median_residual_px": residual.median_px,
                        "p90_residual_px": residual.p90_px,
                        "coherent_fraction": residual.coherent_fraction,
                        "valid_pixels": residual.valid_pixels,
                    })
            medians = [row["median_residual_px"] for row in candidate_measurements]
            steps.append({
                "relative_step": relative,
                "background_valid": estimate.valid,
                "background_confidence": estimate.confidence,
                "num_measured_candidates": len(candidate_measurements),
                "set_max_median_residual_px": max(medians) if medians else None,
                "set_median_median_residual_px": float(np.median(medians)) if medians else None,
                "candidates": candidate_measurements,
            })
        valid_set = [row["set_max_median_residual_px"] for row in steps if row["set_max_median_residual_px"] is not None]
        results.append({
            "episode_id": record["episode_id"], "role": record["role"],
            "action_index": record["action_index"], "status": "measured" if valid_set else "unmeasured",
            "num_seed_candidates": len(seed_masks),
            "window_median_set_max_residual_px": float(np.median(valid_set)) if valid_set else None,
            "steps": steps,
        })
        if (record_index + 1) % 10 == 0:
            print(f"PROGRESS {record_index + 1}/{len(index['records'])}", flush=True)
    payload = {
        "schema_version": 1, "model": args.model,
        "feature_firewall": index["feature_firewall"],
        "warning": "Raw residuals are measurements, not anomaly probabilities.",
        "records": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    measured = [r for r in results if r["status"] == "measured"]
    print(json.dumps({"windows": len(results), "measured": len(measured)}))


if __name__ == "__main__":
    main()
