#!/usr/bin/env python3
"""Label-free sensitivity of gripper-proximity gates to calibration uncertainty."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vla_supervisor.gripper_projection import GripperPixelProjector


def _points(values: list) -> np.ndarray:
    return np.asarray([[np.nan, np.nan] if value is None else value for value in values], dtype=float)


def eligible_candidates(prediction: dict, state: dict, close_frames: list[int], projector: GripperPixelProjector,
                        radius_px: float, uncertainty_px: float, horizon: int) -> set[int]:
    gripper = np.asarray([projector.project(xyz).center_xy for xyz in state["eef_position_xyz"]], dtype=float)
    threshold = radius_px + uncertainty_px
    eligible: set[int] = set()
    for raw_id, raw_track in prediction["centroids_xy"].items():
        track = _points(raw_track)
        for close in close_frames:
            stop = min(len(track), len(gripper), close + horizon + 1)
            if close >= stop:
                continue
            distance = np.linalg.norm(track[close:stop] - gripper[close:stop], axis=1)
            if np.isfinite(distance).any() and float(np.nanmin(distance)) <= threshold:
                eligible.add(int(raw_id))
                break
    return eligible


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--robot-state", type=Path, required=True)
    parser.add_argument("--projection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=40)
    args = parser.parse_args()

    predictions = json.loads(args.predictions.read_text(encoding="utf-8"))["records"]
    features = {r["anonymous_id"]: r for r in json.loads(args.features.read_text(encoding="utf-8"))["records"]}
    states = {r["anonymous_id"]: r for r in json.loads(args.robot_state.read_text(encoding="utf-8"))["records"]}
    projector = GripperPixelProjector.from_fit_file(args.projection)
    p90 = projector.uncertainty_p90_px
    configurations = [(radius, uncertainty) for radius in (20.0, 30.0, 40.0) for uncertainty in (0.0, p90 / 2, p90, 30.0)]
    reference = (30.0, p90)
    sets: dict[tuple[float, float], dict[str, set[int]]] = {}
    for configuration in configurations:
        radius, uncertainty = configuration
        sets[configuration] = {
            row["anonymous_id"]: eligible_candidates(
                row, states[row["anonymous_id"]], features[row["anonymous_id"]]["close_frames"],
                projector, radius, uncertainty, args.horizon,
            )
            for row in predictions
        }

    reference_sets = sets[reference]
    summary = []
    for radius, uncertainty in configurations:
        current = sets[(radius, uncertainty)]
        changed = sum(current[key] != reference_sets[key] for key in reference_sets)
        counts = [len(value) for value in current.values()]
        empty = sum(not value for value in current.values())
        summary.append({
            "base_radius_px": radius,
            "uncertainty_px": uncertainty,
            "effective_radius_px": radius + uncertainty,
            "episodes_changed_vs_reference": changed,
            "empty_episode_count": empty,
            "mean_eligible_candidates": float(np.mean(counts)),
            "max_eligible_candidates": max(counts),
        })
    per_episode = []
    for identifier, reference_set in reference_sets.items():
        variants = {f"r{r:g}_u{u:.3f}": sorted(sets[(r, u)][identifier]) for r, u in configurations}
        unique_sets = {tuple(value) for value in variants.values()}
        per_episode.append({
            "anonymous_id": identifier,
            "reference_eligible_ids": sorted(reference_set),
            "num_unique_eligibility_sets": len(unique_sets),
            "stable_across_grid": len(unique_sets) == 1,
            "variants": variants,
        })
    result = {
        "schema_version": 1,
        "analysis_type": "label_free_projection_gate_sensitivity",
        "reference": {"base_radius_px": reference[0], "uncertainty_px": reference[1]},
        "calibration_loo_p90_px": p90,
        "episodes": len(predictions),
        "stable_episode_count": sum(row["stable_across_grid"] for row in per_episode),
        "summary": summary,
        "records": per_episode,
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("episodes", "stable_episode_count", "calibration_loo_p90_px")}, indent=2))
    for row in summary:
        print(json.dumps(row))


if __name__ == "__main__":
    main()
