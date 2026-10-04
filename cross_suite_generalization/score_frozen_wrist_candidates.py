#!/usr/bin/env python3
"""Apply the frozen wrist physical candidate protocol without outcome labels."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from cross_suite_generalization.analyze_wrist_attachment_changepoint import (
    log_drop,
    motion_features,
)
from cross_suite_generalization.evaluate_wrist_physical_fusion import (
    distance_to_box,
    rank_fraction,
    visibility,
)
from vla_supervisor.wrist_physical_evidence import (
    PlattCalibration,
    WristCandidateScore,
    WristSelectionConfig,
    select_wrist_candidates,
)


def change_scores(record: dict, image_size: int, post_settle: int = 5,
                  post_window: int = 25) -> dict[int, float]:
    scores = {}
    for candidate_key, pre_centroids in record["pre_centroids_xy"].items():
        pre = motion_features(
            pre_centroids[:-1], record["pre_area_fraction"][candidate_key][:-1], image_size
        )
        post = motion_features(
            record["post_centroids_xy"][candidate_key][post_settle:post_settle + post_window],
            record["post_area_fraction"][candidate_key][post_settle:post_settle + post_window],
            image_size,
        )
        scores[int(candidate_key)] = (
            log_drop(pre["speed_median"], post["speed_median"])
            + .5 * log_drop(pre["dispersion_p90"], post["dispersion_p90"])
            + .25 * log_drop(pre["area_delta_median"], post["area_delta_median"])
        )
    return scores


def validate_record(record: dict, post_settle: int = 5,
                    post_window: int = 25) -> None:
    """Reject incomplete bidirectional tracks instead of silently rescoring them."""
    if record.get("close_frame") is None or not record.get("boxes_xywh"):
        return
    expected = {str(index) for index in range(1, len(record["boxes_xywh"]) + 1)}
    for field in (
        "pre_centroids_xy", "pre_area_fraction",
        "post_centroids_xy", "post_area_fraction",
    ):
        actual = set(record.get(field, {}))
        if actual != expected:
            raise ValueError(
                f"{record['anonymous_id']}: {field} candidate ids {sorted(actual)} "
                f"do not match {sorted(expected)}"
            )
    for candidate_id in expected:
        if len(record["pre_centroids_xy"][candidate_id]) < 2:
            raise ValueError(
                f"{record['anonymous_id']}: candidate {candidate_id} lacks pre-close history"
            )
        if len(record["post_centroids_xy"][candidate_id]) < post_settle + post_window:
            raise ValueError(
                f"{record['anonymous_id']}: candidate {candidate_id} has only "
                f"{len(record['post_centroids_xy'][candidate_id])} post-close frames; "
                f"need {post_settle + post_window}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = json.loads(args.predictions.read_text(encoding="utf-8"))["records"]
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    geometry = protocol["pinch_center_xy"]
    image_size = int(protocol["image_size"])
    selection = protocol["set_valued_selection"]
    frozen = protocol["platt_calibration"]
    calibration = PlattCalibration(
        float(frozen["normalization_mean"]),
        float(frozen["normalization_scale"]),
        float(frozen["intercept"]),
        float(frozen["coefficient"]),
    )
    config = WristSelectionConfig(
        score_margin=float(selection["score_margin"]),
        minimum_visibility=float(selection["minimum_visibility"]),
    )
    output_records = []
    for record in records:
        validate_record(record)
        if record.get("close_frame") is None or not record.get("boxes_xywh"):
            output_records.append({
                "anonymous_id": record["anonymous_id"],
                "close_frame": record.get("close_frame"),
                "state": "no_candidate",
                "candidate_set": [],
                "controlled_object_candidate_probability": None,
                "candidates": [],
            })
            continue
        raw_change = change_scores(record, image_size)
        ranks = rank_fraction(raw_change)
        candidates = []
        selector_inputs = []
        for candidate_id, box in enumerate(record["boxes_xywh"], 1):
            key = str(candidate_id)
            distance = distance_to_box(tuple(map(float, geometry)), box)
            proximity = math.exp(-distance / 32.0)
            visible = visibility(record["post_centroids_xy"][key])
            left, top, width, height = map(float, box)
            robot_band = float(width >= .80 * image_size and top >= .72 * image_size)
            full_frame = float(width * height >= .70 * image_size * image_size)
            score = (
                .55 * ranks[candidate_id] + .25 * proximity + .20 * visible
                - .35 * robot_band - .20 * full_frame
            )
            candidates.append({
                "candidate_id": candidate_id,
                "combined_change_score": raw_change[candidate_id],
                "change_rank_fraction": ranks[candidate_id],
                "pinch_box_distance_px": distance,
                "pinch_proximity": proximity,
                "postclose_visibility": visible,
                "robot_bottom_band_penalty": robot_band,
                "near_full_frame_penalty": full_frame,
                "physical_fusion_score": score,
            })
            selector_inputs.append(WristCandidateScore(str(candidate_id), score, visible))
        selected = select_wrist_candidates(selector_inputs, config, calibration)
        output_records.append({
            "anonymous_id": record["anonymous_id"],
            "close_frame": int(record["close_frame"]),
            "state": selected.state.value,
            "candidate_set": [int(value) for value in selected.candidate_ids],
            "top_score": selected.top_score,
            "runner_up_gap": selected.runner_up_gap,
            # This is probability that a unique mask is an acceptable physical
            # controlled-object candidate. It is not grasp success probability.
            "controlled_object_candidate_probability": selected.calibrated_candidate_probability,
            "passes_frozen_candidate_gate": (
                selected.calibrated_candidate_probability is not None
                and selected.calibrated_candidate_probability >= float(frozen["attachment_gate"])
            ),
            "candidates": candidates,
        })
    counts = {}
    for record in output_records:
        counts[record["state"]] = counts.get(record["state"], 0) + 1
    result = {
        "schema_version": 1,
        "role": "outcome-blind frozen candidate scoring",
        "probability_semantics": (
            "acceptable controlled-object mask candidate; not physical grasp success"
        ),
        "protocol": str(args.protocol),
        "records": output_records,
        "summary": {
            "episodes": len(output_records),
            "state_counts": counts,
            "unique_above_frozen_gate": sum(
                bool(record.get("passes_frozen_candidate_gate")) for record in output_records
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
