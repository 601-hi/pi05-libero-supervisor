#!/usr/bin/env python3
"""Evaluate a conservative progress-only articulation watchdog."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from vla_supervisor.articulation_progress import (
    ArticulationMeasurement,
    ArticulationProgressConfig,
    ArticulationProgressWatchdog,
)


def classify(area: list[float]) -> dict:
    config = ArticulationProgressConfig(
        calibration_id="task0-close-progress-v1",
        initial_grace_actions=80,
        active_gain=0.15,
        plateau_window_actions=30,
        refresh_gain=0.03,
        regression_drop=0.25,
        regression_confirmations=5,
        smoothing_samples=5,
        minimum_confidence=0.70,
    )
    watchdog = ArticulationProgressWatchdog(config)
    decisions = []
    for action_index, value in enumerate(area):
        decisions.append(watchdog.update(ArticulationMeasurement(
            action_index=action_index, area_fraction=float(value), confidence=.90,
            observable=True, identity_reliable=True, calibrated=True,
            calibration_id=config.calibration_id, relation_id="top_drawer_closed",
            predicate="Close", provenance="frozen_sam2_replay")))
    actionable = [d for d in decisions if d.state in {"goal_no_progress", "goal_regression"}]
    first = actionable[0] if actionable else None
    return {
        "event": first.state if first else "continue",
        "trigger_frame": first.action_index if first else None,
        "terminal_progress": decisions[-1].progress,
        "best_progress": decisions[-1].best_progress,
        "parameters": config.__dict__,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", type=Path, nargs="+", required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    records = []
    for path in args.inputs:
        data = json.loads(path.read_text(encoding="utf-8"))
        for record in data["records"]:
            records.append({"identity": record["identity"], **classify(record["area_fraction"])})
    result = {
        "schema_version": 1,
        "interpretation": "progress-only; never authorizes satisfied/complete",
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
