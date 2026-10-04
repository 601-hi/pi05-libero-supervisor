#!/usr/bin/env python3
"""Build outcome/risk candidates without treating interventions as ground truth faults.

The script consumes the episode-level audit JSON.  Natural successful episodes form
task-conditioned reference distributions.  Failures are task-critical labels; high
motion-cost scores are *review candidates*, not automatic safety labels.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


METRICS = (
    "steps",
    "eef_path_m",
    "joint_path_rad",
    "max_abs_joint_acceleration",
    "max_abs_joint_jerk",
    "p95_abs_joint_jerk",
    "rms_joint_jerk",
    "high_jerk_joint_events_gt100",
    "gripper_command_reversals",
    "joint_velocity_reversals",
    "max_low_progress_run",
)
EFFICIENCY_METRICS = {"steps", "eef_path_m", "joint_path_rad", "max_low_progress_run"}
SMOOTHNESS_METRICS = set(METRICS) - EFFICIENCY_METRICS


def infer_suite(path: str) -> str:
    value = path.lower()
    for suite in ("libero_90", "libero_10", "libero_goal", "libero_object", "libero_spatial"):
        if suite in value:
            return suite
    if "spatial" in value:
        return "libero_spatial"
    return "unknown"


def robust_location_scale(values: list[float]) -> tuple[float, float]:
    x = np.asarray(values, dtype=float)
    median = float(np.median(x))
    mad = float(np.median(np.abs(x - median)))
    # 1.4826 makes MAD comparable to standard deviation under a Gaussian model.
    scale = max(1.4826 * mad, float(np.quantile(x, 0.75) - np.quantile(x, 0.25)) / 1.349,
                float(np.std(x)), 1e-9)
    return median, scale


def auc(labels: list[int], scores: list[float]) -> float | None:
    y = np.asarray(labels, int); s = np.asarray(scores, float)
    pos = s[y == 1]; neg = s[y == 0]
    if not len(pos) or not len(neg):
        return None
    # Pairwise definition handles ties explicitly and avoids an sklearn dependency.
    return float(((pos[:, None] > neg[None, :]).mean() +
                  0.5 * (pos[:, None] == neg[None, :]).mean()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-reference-episodes", type=int, default=5)
    args = parser.parse_args()

    payload = json.loads(args.audit.read_text(encoding="utf-8"))
    episodes = payload["episodes"]
    for row in episodes:
        row["suite"] = infer_suite(row["path"])

    # Do not let injected runs define normality, even when they eventually succeed.
    references_raw: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in episodes:
        if row["natural"] and row["success"]:
            references_raw[(row["suite"], row["task_id"])].append(row)
    # Fixed-noise reproducibility runs may be exactly duplicated.  They are useful
    # paired experiments but are not independent calibration evidence.
    references = {}
    for key, values in references_raw.items():
        unique = {}
        for row in values:
            signature = tuple(round(float(row[metric]), 12) for metric in METRICS)
            unique.setdefault(signature, row)
        references[key] = list(unique.values())

    calibrated = []
    for row in episodes:
        key = (row["suite"], row["task_id"])
        ref = references.get(key, [])
        out = dict(row)
        out["task_critical"] = not row["success"]
        out["reference_key"] = {"suite": key[0], "task_id": key[1]}
        out["reference_raw_count"] = len(references_raw.get(key, []))
        out["reference_count"] = len(ref)
        out["motion_quality_reviewable"] = len(ref) >= args.min_reference_episodes
        z = {}
        q95_exceeded = []
        if out["motion_quality_reviewable"]:
            for metric in METRICS:
                values = [float(x[metric]) for x in ref]
                location, scale = robust_location_scale(values)
                z[metric] = (float(row[metric]) - location) / scale
                if float(row[metric]) > float(np.quantile(values, 0.95)):
                    q95_exceeded.append(metric)
        out["task_conditioned_robust_z"] = z
        # The cap prevents zero-spread reference metrics from masquerading as
        # billion-sigma evidence; raw metric tails remain available above.
        out["motion_quality_score"] = min(max(z.values()), 20.0) if z else None
        out["task_conditioned_q95_exceeded"] = q95_exceeded
        efficiency_tails = sum(x in EFFICIENCY_METRICS for x in q95_exceeded)
        smoothness_tails = sum(x in SMOOTHNESS_METRICS for x in q95_exceeded)
        out["efficiency_q95_exceedance_count"] = efficiency_tails
        out["smoothness_q95_exceedance_count"] = smoothness_tails
        # Correlated jerk summaries belong to one evidence family and must not
        # each act as an independent vote.  This remains a review rule, not an alarm.
        # A flag asks for inspection; it is deliberately not called an error or hazard.
        out["motion_quality_review_candidate"] = efficiency_tails >= 2 or smoothness_tails >= 3
        calibrated.append(out)

    natural = [x for x in calibrated if x["natural"] and x["motion_quality_score"] is not None]
    labels = [int(x["task_critical"]) for x in natural]
    scores = [float(x["motion_quality_score"]) for x in natural]
    summary = {
        "episodes": len(calibrated),
        "reference_groups": len(references),
        "reference_raw_episodes": sum(map(len, references_raw.values())),
        "reference_unique_signatures": sum(map(len, references.values())),
        "reviewable_episodes": sum(x["motion_quality_reviewable"] for x in calibrated),
        "natural_reviewable_episodes": len(natural),
        "natural_reviewable_failures": sum(labels),
        "motion_quality_score_failure_auc": auc(labels, scores),
        "natural_success_review_candidates": sum(
            x["natural"] and x["success"] and x["motion_quality_review_candidate"] for x in calibrated
        ),
        "natural_failure_review_candidates": sum(
            x["natural"] and not x["success"] and x["motion_quality_review_candidate"] for x in calibrated
        ),
        "unreviewable_task_critical_episodes": sum(
            x["task_critical"] and not x["motion_quality_reviewable"] for x in calibrated
        ),
        "label_policy": {
            "intervention_is_error_label": False,
            "task_failure_is_task_critical": True,
            "motion_outlier_requires_review": True,
            "hardware_safety_requires_current_torque_or_limits": True,
        },
    }
    result = {"summary": summary, "episodes": calibrated}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
