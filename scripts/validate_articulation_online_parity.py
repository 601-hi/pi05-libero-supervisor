#!/usr/bin/env python3
"""GPU gate: compare online image-prompt tracking with frozen video tracking."""
from __future__ import annotations

import argparse
import base64
from io import BytesIO
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_articulation_vision_sidecar import (
    DinoSam2Backend, StreamingDinoSam2Backend)
from vla_supervisor.articulation_progress import (
    ArticulationMeasurement, ArticulationProgressConfig,
    ArticulationProgressWatchdog)


def encode(image):
    buffer = BytesIO()
    Image.fromarray(image.astype(np.uint8), mode="RGB").save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def final_state(areas, calibration_id):
    watchdog = ArticulationProgressWatchdog(ArticulationProgressConfig(
        calibration_id=calibration_id))
    state, trigger = "unknown", None
    for index, area in enumerate(areas):
        decision = watchdog.update(ArticulationMeasurement(
            action_index=index, area_fraction=float(area), confidence=.99,
            observable=True, identity_reliable=True, calibrated=True,
            calibration_id=calibration_id, relation_id="top_drawer_closed",
            predicate="Close", provenance="parity"))
        state = decision.state
        if trigger is None and state in {"goal_no_progress", "goal_regression"}:
            trigger = index
    return state, trigger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--query", default="open top drawer")
    parser.add_argument("--dino-model", default="IDEA-Research/grounding-dino-base")
    parser.add_argument("--sam-model", default="facebook/sam2.1-hiera-small")
    parser.add_argument("--tracking-backend", choices=("streaming_video", "image_prompt"),
                        default="streaming_video")
    parser.add_argument("--minimum-correlation", type=float, default=.85)
    parser.add_argument("--maximum-progress-mae", type=float, default=.10)
    args = parser.parse_args()
    calibration_id = "libero90-task0-close-area-v1"
    backend_class = (StreamingDinoSam2Backend
                     if args.tracking_backend == "streaming_video"
                     else DinoSam2Backend)
    backend = backend_class(
        query=args.query, dino_model=args.dino_model,
        sam_model=args.sam_model, calibrated=False)
    records = []
    for source in args.predictions:
        payload = json.loads(source.read_text(encoding="utf-8"))
        for frozen in payload["records"]:
            backend.reset()
            with np.load(frozen["sidecar"], mmap_mode="r") as data:
                frames = np.asarray(data["agent_images"], dtype=np.uint8)
            online = []
            valid = True
            for index, frame in enumerate(frames):
                measurement = backend.observe({
                    "action_index": index,
                    "agent_image_base64": encode(frame),
                    "anchor_box_xyxy": frozen["anchor_box_xyxy"] if index == 0 else None,
                })
                area = measurement.get("area_fraction")
                if area is None or not measurement.get("identity_reliable", False):
                    valid = False; break
                online.append(float(area))
            frozen_area = np.asarray(frozen["area_fraction"], float)
            online_area = np.asarray(online, float)
            comparable = valid and len(online_area) == len(frozen_area)
            if comparable:
                frozen_progress = 1.0 - frozen_area / frozen_area[0]
                online_progress = 1.0 - online_area / online_area[0]
                mae = float(np.mean(np.abs(frozen_progress - online_progress)))
                if np.std(frozen_progress) < 1e-8 or np.std(online_progress) < 1e-8:
                    correlation = 1.0 if mae <= args.maximum_progress_mae else 0.0
                else:
                    correlation = float(np.corrcoef(frozen_progress, online_progress)[0, 1])
                frozen_state = final_state(frozen_area, calibration_id)
                online_state = final_state(online_area, calibration_id)
            else:
                mae, correlation = None, None
                frozen_state = final_state(frozen_area, calibration_id)
                online_state = ("unknown", None)
            passed = bool(comparable and mae <= args.maximum_progress_mae
                          and correlation >= args.minimum_correlation
                          and online_state[0] == frozen_state[0])
            records.append({
                "identity": frozen["identity"], "source": str(source),
                "frames": len(frames), "all_frames_reliable": valid,
                "progress_mae": mae, "progress_correlation": correlation,
                "frozen_final_state": frozen_state[0],
                "frozen_trigger": frozen_state[1],
                "online_final_state": online_state[0],
                "online_trigger": online_state[1], "passed": passed,
            })
            print(json.dumps(records[-1], ensure_ascii=False), flush=True)
    result = {
        "schema_version": 1,
        "status": "passed" if records and all(x["passed"] for x in records) else "failed",
        "calibrated_after_gate": bool(records and all(x["passed"] for x in records)),
        "thresholds": {"minimum_correlation": args.minimum_correlation,
                       "maximum_progress_mae": args.maximum_progress_mae},
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps({"status": result["status"], "records": len(records)}))
    raise SystemExit(0 if result["status"] == "passed" else 2)


if __name__ == "__main__":
    main()
