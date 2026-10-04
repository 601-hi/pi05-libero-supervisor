#!/usr/bin/env python3
"""Fuse event-level fixed motion with wrist candidate evidence conservatively.

The fixed view can support that something moved after closure, but it cannot
identify a wrist-mask ID.  The wrist view can rank mask candidates, but a
unique calibrated candidate alone does not establish contact.  This module
keeps those roles separate and emits explicit abstention states.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
from pathlib import Path


def sigmoid(value: float) -> float:
    value = max(-40.0, min(40.0, value))
    return 1.0 / (1.0 + math.exp(-value))


def evidence_states(group_count: int, candidate_count: int, probability: float | None, gate: float) -> tuple[str, str, str]:
    if group_count == 0:
        fixed_state = "no_motion"
    elif group_count == 1:
        fixed_state = "single_motion_pattern"
    else:
        fixed_state = "multiple_motion_patterns"
    if candidate_count == 0:
        wrist_state = "no_candidate"
    elif candidate_count > 1:
        wrist_state = "ambiguous"
    elif probability is None or probability < gate:
        wrist_state = "unique_low_confidence"
    else:
        wrist_state = "unique_high_confidence"

    if fixed_state == "no_motion":
        joint_state = "no_fixed_motion_evidence"
    elif wrist_state == "no_candidate":
        joint_state = "wrist_no_candidate"
    elif wrist_state == "ambiguous":
        joint_state = "wrist_ambiguous"
    elif wrist_state == "unique_low_confidence":
        joint_state = "unique_wrist_low_confidence"
    elif fixed_state == "multiple_motion_patterns":
        joint_state = "possible_control_crossview_ambiguous"
    else:
        joint_state = "possible_control"
    return fixed_state, wrist_state, joint_state


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixed-motion", type=Path, required=True)
    parser.add_argument("--wrist-fusion", type=Path, required=True)
    parser.add_argument("--frozen-protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    fixed = {
        row["anonymous_id"]: row
        for row in json.loads(args.fixed_motion.read_text(encoding="utf-8"))["records"]
    }
    wrist = json.loads(args.wrist_fusion.read_text(encoding="utf-8"))["episodes"]
    protocol = json.loads(args.frozen_protocol.read_text(encoding="utf-8"))
    selection = protocol["set_valued_selection"]
    calibration = protocol["platt_calibration"]
    margin = float(selection["score_margin"])
    minimum_visibility = float(selection["minimum_visibility"])

    rows = []
    for episode in wrist:
        identifier = episode["anonymous_id"]
        if identifier not in fixed or not fixed[identifier]["events"]:
            raise RuntimeError(f"Missing fixed first-close event for {identifier}")
        event = fixed[identifier]["events"][0]
        groups = event["co_motion_groups"]
        eligible = [
            candidate for candidate in episode["candidates"]
            if float(candidate["postclose_visibility"]) >= minimum_visibility
        ]
        best = max((float(candidate["physical_fusion_score"]) for candidate in eligible), default=None)
        candidate_set = [] if best is None else [
            int(candidate["candidate_id"]) for candidate in eligible
            if float(candidate["physical_fusion_score"]) >= best - margin
        ]
        probability = None
        if len(candidate_set) == 1:
            normalized = (best - float(calibration["normalization_mean"])) / float(calibration["normalization_scale"])
            probability = sigmoid(float(calibration["intercept"]) + float(calibration["coefficient"]) * normalized)

        fixed_state, wrist_state, state = evidence_states(
            len(groups), len(candidate_set), probability, float(calibration["attachment_gate"])
        )
        rows.append({
            "anonymous_id": identifier,
            "close_frame": int(event["close_frame"]),
            "fixed_motion_group_count": len(groups),
            "fixed_motion_groups": groups,
            "fixed_view_ambiguous": len(groups) > 1,
            "fixed_evidence_state": fixed_state,
            "wrist_candidate_set": candidate_set,
            "unique_candidate_probability": probability,
            "wrist_evidence_state": wrist_state,
            "state": state,
        })

    counts = collections.Counter(row["state"] for row in rows)
    result = {
        "schema_version": 1,
        "interpretation": "Coverage audit without outcome labels; possible_control is not success or semantic correctness.",
        "forbidden_claims": [
            "fixed motion group and wrist candidate share an object identity",
            "possible_control means task success",
            "unique wrist score can veto an execution anomaly",
        ],
        "frozen_parameters": {
            "score_margin": margin,
            "minimum_visibility": minimum_visibility,
            "attachment_gate": float(calibration["attachment_gate"]),
        },
        "episodes": len(rows),
        "state_counts": dict(counts),
        "records": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"episodes": len(rows), "state_counts": dict(counts)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
