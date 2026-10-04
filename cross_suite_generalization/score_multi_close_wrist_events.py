"""Apply the frozen wrist scorer to multi-close events with censoring safeguards."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from cross_suite_generalization.score_frozen_wrist_candidates import change_scores
from cross_suite_generalization.evaluate_wrist_physical_fusion import distance_to_box, rank_fraction, visibility
from vla_supervisor.wrist_physical_evidence import (
    PlattCalibration, WristCandidateScore, WristSelectionConfig, select_wrist_candidates,
)


POST_SETTLE = 5
POST_WINDOW = 25


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = json.loads(args.predictions.read_text(encoding="utf-8"))["records"]
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    geometry, image_size = protocol["pinch_center_xy"], int(protocol["image_size"])
    selection, frozen = protocol["set_valued_selection"], protocol["platt_calibration"]
    calibration = PlattCalibration(
        float(frozen["normalization_mean"]), float(frozen["normalization_scale"]),
        float(frozen["intercept"]), float(frozen["coefficient"]),
    )
    config = WristSelectionConfig(
        score_margin=float(selection["score_margin"]),
        minimum_visibility=float(selection["minimum_visibility"]),
    )
    outputs = []
    for record in records:
        base = {
            "event_id": record["event_id"], "anonymous_id": record["anonymous_id"],
            "close_ordinal": record["close_ordinal"], "close_frame": record["close_frame"],
        }
        if not record.get("boxes_xywh"):
            outputs.append({**base, "state": "no_candidate", "candidate_set": [],
                            "passes_frozen_candidate_gate": False, "candidates": []})
            continue
        post_lengths = [len(values) for values in record["post_centroids_xy"].values()]
        if min(post_lengths, default=0) < POST_SETTLE + POST_WINDOW:
            outputs.append({
                **base, "state": "insufficient_post_window", "candidate_set": [],
                "available_post_frames": min(post_lengths, default=0),
                "required_post_frames": POST_SETTLE + POST_WINDOW,
                "passes_frozen_candidate_gate": False, "candidates": [],
            })
            continue
        raw_change, candidates, selector_inputs = change_scores(record, image_size), [], []
        ranks = rank_fraction(raw_change)
        for candidate_id, box in enumerate(record["boxes_xywh"], 1):
            key = str(candidate_id)
            distance = distance_to_box(tuple(map(float, geometry)), box)
            proximity, visible = math.exp(-distance / 32.0), visibility(record["post_centroids_xy"][key])
            _, top, width, height = map(float, box)
            robot_band = float(width >= .80 * image_size and top >= .72 * image_size)
            full_frame = float(width * height >= .70 * image_size * image_size)
            score = .55 * ranks[candidate_id] + .25 * proximity + .20 * visible - .35 * robot_band - .20 * full_frame
            candidates.append({
                "candidate_id": candidate_id, "combined_change_score": raw_change[candidate_id],
                "change_rank_fraction": ranks[candidate_id], "pinch_box_distance_px": distance,
                "postclose_visibility": visible, "physical_fusion_score": score,
            })
            selector_inputs.append(WristCandidateScore(str(candidate_id), score, visible))
        selected = select_wrist_candidates(selector_inputs, config, calibration)
        probability = selected.calibrated_candidate_probability
        outputs.append({
            **base, "state": selected.state.value,
            "candidate_set": [int(value) for value in selected.candidate_ids],
            "top_score": selected.top_score, "runner_up_gap": selected.runner_up_gap,
            "controlled_object_candidate_probability": probability,
            "passes_frozen_candidate_gate": probability is not None and probability >= float(frozen["attachment_gate"]),
            "candidates": candidates,
        })
    counts = {}
    for row in outputs:
        counts[row["state"]] = counts.get(row["state"], 0) + 1
    result = {
        "schema_version": 1,
        "probability_semantics": "acceptable controlled-object mask candidate; not grasp success",
        "censoring_rule": f"requires {POST_SETTLE + POST_WINDOW} post-close frames; shorter tail events are not rescored",
        "summary": {"events": len(outputs), "state_counts": counts,
                    "unique_above_frozen_gate": sum(r["passes_frozen_candidate_gate"] for r in outputs)},
        "records": outputs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
