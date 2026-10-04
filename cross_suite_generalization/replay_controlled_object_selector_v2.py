#!/usr/bin/env python3
"""Replay the online controlled-object selector from outcome-free Wave1 artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vla_supervisor.controlled_object_selector_v2 import CandidateControlEvidence, ControlledObjectSelectorV2
from vla_supervisor.gripper_projection import GripperPixelProjector


def rolling_motion_likeness(track_values: list, gripper: np.ndarray, window: int = 15) -> np.ndarray:
    track = np.asarray([[np.nan, np.nan] if point is None else point for point in track_values], dtype=float)
    result = np.zeros(min(len(track), len(gripper)), dtype=float)
    candidate_delta = np.diff(track, axis=0)
    gripper_delta = np.diff(gripper, axis=0)
    for frame in range(1, len(result)):
        start = max(0, frame - window)
        left, right = candidate_delta[start:frame], gripper_delta[start:frame]
        valid = np.all(np.isfinite(left), axis=1) & np.all(np.isfinite(right), axis=1)
        valid &= (np.linalg.norm(left, axis=1) > 0.25) & (np.linalg.norm(right, axis=1) > 0.25)
        if valid.sum() >= 3:
            cosine = np.sum(left[valid] * right[valid], axis=1) / (
                np.linalg.norm(left[valid], axis=1) * np.linalg.norm(right[valid], axis=1) + 1e-9
            )
            result[frame] = max(0.0, float(np.median(cosine)))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--robot-state", type=Path, required=True)
    parser.add_argument("--projection", type=Path, required=True)
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--proximity-radius-px", type=float, default=30.0)
    parser.add_argument("--uncertainty-scale", type=float, default=1.0)
    parser.add_argument("--preclose-robot-motion-threshold", type=float)
    args = parser.parse_args()

    predictions = json.loads(args.predictions.read_text(encoding="utf-8"))["records"]
    features = {r["anonymous_id"]: r for r in json.loads(args.features.read_text(encoding="utf-8"))["records"]}
    states = {r["anonymous_id"]: r for r in json.loads(args.robot_state.read_text(encoding="utf-8"))["records"]}
    projector = GripperPixelProjector.from_fit_file(args.projection)
    truth = None
    if args.annotations:
        truth = {r["anonymous_id"]: r for r in json.loads(args.annotations.read_text(encoding="utf-8"))["records"]}

    output = []
    status_counts: dict[str, int] = {}
    for prediction in predictions:
        identifier = prediction["anonymous_id"]
        feature = features[identifier]
        state = states[identifier]
        selector = ControlledObjectSelectorV2(
            proximity_radius_px=args.proximity_radius_px,
            maximum_preclose_robot_motion_likeness=(
                args.preclose_robot_motion_threshold
                if args.preclose_robot_motion_threshold is not None else 1.0
            ),
        )
        projected_track = np.asarray([projector.project(xyz).center_xy for xyz in state["eef_position_xyz"]], dtype=float)
        robot_likeness = {}
        close_set = set(feature["close_frames"])
        for candidate_id, track in prediction["centroids_xy"].items():
            rolling = rolling_motion_likeness(track, projected_track)
            frozen = np.zeros_like(rolling)
            current = 0.0
            for frame_index in range(len(frozen)):
                if frame_index in close_set:
                    current = rolling[frame_index]
                frozen[frame_index] = current
            robot_likeness[candidate_id] = frozen
        frames = []
        first_lock = None
        length = min(len(state["eef_position_xyz"]), len(state["gripper_command"]))
        for frame_index in range(length):
            projected = projector.project(state["eef_position_xyz"][frame_index])
            candidates = {}
            for candidate_id, track in prediction["centroids_xy"].items():
                point = track[frame_index] if frame_index < len(track) else None
                residuals = feature["candidates"][candidate_id].get("background_residual_sequence_normalized", [])
                residual = residuals[frame_index] if frame_index < len(residuals) else None
                likeness = (
                    robot_likeness[candidate_id][frame_index]
                    if args.preclose_robot_motion_threshold is not None and frame_index < len(robot_likeness[candidate_id])
                    else 0.0
                )
                candidates[int(candidate_id)] = CandidateControlEvidence(
                    point, residual, preclose_robot_motion_likeness=float(likeness)
                )
            background_confidence = feature.get("background_confidence_sequence", [])
            confidence = background_confidence[frame_index] if frame_index < len(background_confidence) else 0.0
            moving = bool(state["eef_translation_norm"][frame_index] > 1e-5)
            decision = selector.update(
                gripper_command=float(state["gripper_command"][frame_index]),
                gripper_moving=moving,
                gripper_xy=projected.center_xy,
                projection_uncertainty_px=projected.uncertainty_p90_px * args.uncertainty_scale,
                candidates=candidates,
                background_confidence=float(confidence),
            )
            status_counts[decision.state] = status_counts.get(decision.state, 0) + 1
            if first_lock is None and decision.controlled_object_id is not None:
                first_lock = {"frame": frame_index, "candidate_id": decision.controlled_object_id}
            frames.append({
                "frame": frame_index,
                "state": decision.state,
                "controlled_object_id": decision.controlled_object_id,
                "leading_object_id": decision.leading_object_id,
                "score_margin": decision.score_margin,
                "confirmation_streak": decision.confirmation_streak,
                "evidence_valid": decision.evidence_valid,
            })
        evaluation = None
        if truth is not None:
            label = truth[identifier]
            if not label["target_mask_observable"]:
                evaluation = "unobservable"
            elif first_lock is None:
                evaluation = "unknown"
            elif first_lock["candidate_id"] in set(label["acceptable_target_mask_ids"]):
                evaluation = "correct"
            else:
                evaluation = "wrong"
        output.append({"anonymous_id": identifier, "first_lock": first_lock, "evaluation": evaluation, "frames": frames})

    result = {
        "schema_version": 1,
        "parameters": {
            "proximity_radius_px": args.proximity_radius_px,
            "projection_uncertainty_scale": args.uncertainty_scale,
            "projection_uncertainty_p90_px": projector.uncertainty_p90_px,
            "preclose_robot_motion_threshold": args.preclose_robot_motion_threshold,
            "robot_overlap_evidence_available": False,
        },
        "episodes": len(output),
        "episodes_with_lock": sum(row["first_lock"] is not None for row in output),
        "state_counts": status_counts,
        "evaluation_counts": ({key: sum(row["evaluation"] == key for row in output) for key in ("correct", "wrong", "unknown", "unobservable")} if truth else None),
        "records": output,
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("episodes", "episodes_with_lock", "state_counts", "evaluation_counts")}, indent=2))


if __name__ == "__main__":
    main()
