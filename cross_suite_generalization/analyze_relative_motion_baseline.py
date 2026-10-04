"""Mechanism audit for set-level relative motion; never selects a threshold."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from vla_supervisor.adaptive_motion_baseline import AdaptiveMotionBaseline


def frame_max(path: Path) -> list[tuple[int, float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    grouped: dict[int, list[float]] = {}
    for candidate in payload["candidates"]:
        for step in candidate["per_step"]:
            value = step["median_residual_px"]
            if value is not None:
                grouped.setdefault(int(step["action_index"]), []).append(float(value))
    return [(index, max(values)) for index, values in sorted(grouped.items())]


def percentiles(values: list[float]) -> dict[str, float]:
    return {
        "p10": float(np.percentile(values, 10)),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reference = frame_max(args.reference)
    evaluation = frame_max(args.evaluation)
    baseline = AdaptiveMotionBaseline(
        window=50, minimum_samples=12, minimum_scale=.02,
        minimum_background_confidence=0,
    )
    for _, value in reference:
        baseline.observe(value, background_confidence=1, allow_reference_update=True)
    scored = []
    for action_index, value in evaluation:
        evidence = baseline.observe(
            value, background_confidence=1, allow_reference_update=False
        )
        scored.append({
            "action_index": action_index,
            "set_max_median_residual_px": value,
            "motion_drop_z": evidence.motion_drop_z,
        })
    drop = [row["motion_drop_z"] for row in scored if row["motion_drop_z"] is not None]
    result = {
        "schema_version": 1,
        "analysis_role": "mechanism audit only; no alarm threshold selected",
        "aggregation": "maximum candidate median residual per action",
        "reference": {"steps": len(reference), **percentiles([value for _, value in reference])},
        "evaluation": {"steps": len(evaluation), **percentiles([value for _, value in evaluation])},
        "motion_drop_z": {
            **percentiles(drop),
            "fraction_gt_2": float(np.mean(np.asarray(drop) > 2)),
            "fraction_gt_3": float(np.mean(np.asarray(drop) > 3)),
        },
        "scored_steps": scored,
    }
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps({key: result[key] for key in ("reference", "evaluation", "motion_drop_z")}))


if __name__ == "__main__":
    main()
