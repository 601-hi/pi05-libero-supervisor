"""Audit task-agnostic dual-expert transfer on paired object-outcome pilots.

This audit deliberately distinguishes a detector that alarms during a fault
from one whose first alarm happened before the intervention.  It also reports
paired score changes, which cancel the shared pre-intervention suite shift but
are diagnostic only and are not a deployable decision rule.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _indices_for_source(data: np.lib.npyio.NpzFile, source_id: int) -> np.ndarray:
    prefix = f"source{source_id}:"
    indices = np.asarray(
        [i for i, episode in enumerate(data["episode_id"]) if str(episode).startswith(prefix)],
        dtype=np.int64,
    )
    if not len(indices):
        raise ValueError(f"source {source_id} is absent")
    return indices[np.argsort(data["action_index"][indices])]


def _segment(values: np.ndarray, start: int, stop: int) -> dict[str, float | int] | None:
    selected = values[start:stop]
    if not len(selected):
        return None
    return {
        "count": int(len(selected)),
        "median": float(np.median(selected)),
        "minimum": float(np.min(selected)),
        "maximum": float(np.max(selected)),
    }


def audit_pair(
    data: np.lib.npyio.NpzFile,
    name: str,
    normal_source: int,
    fault_source: int,
) -> dict[str, object]:
    normal_indices = _indices_for_source(data, normal_source)
    fault_indices = _indices_for_source(data, fault_source)
    common = min(len(normal_indices), len(fault_indices))
    normal_indices = normal_indices[:common]
    fault_indices = fault_indices[:common]

    active_positions = np.flatnonzero(data["abnormal"][fault_indices].astype(bool))
    if not len(active_positions):
        raise ValueError(f"{name}: fault source has no active disturbance rows")
    onset = int(active_positions[0])
    stop = int(active_positions[-1]) + 1

    # Positive values mean the abnormal expert is more supportive than the
    # normal expert.  The paired delta is diagnostic, not an online feature.
    normal_ratio = data["abnormal_logp"][normal_indices] - data["normal_logp"][normal_indices]
    fault_ratio = data["abnormal_logp"][fault_indices] - data["normal_logp"][fault_indices]
    paired_delta = fault_ratio - normal_ratio

    return {
        "name": name,
        "common_steps": int(common),
        "onset": onset,
        "active_stop_exclusive": stop,
        "pre_ratio_exact_match": bool(np.array_equal(normal_ratio[:onset], fault_ratio[:onset])),
        "normal_ratio": {
            "pre": _segment(normal_ratio, max(0, onset - 10), onset),
            "active": _segment(normal_ratio, onset, stop),
            "post": _segment(normal_ratio, stop, min(common, stop + 15)),
        },
        "fault_ratio": {
            "pre": _segment(fault_ratio, max(0, onset - 10), onset),
            "active": _segment(fault_ratio, onset, stop),
            "post": _segment(fault_ratio, stop, min(common, stop + 15)),
        },
        "paired_delta": {
            "pre": _segment(paired_delta, max(0, onset - 10), onset),
            "active": _segment(paired_delta, onset, stop),
            "post": _segment(paired_delta, stop, min(common, stop + 15)),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--fusion", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    data = np.load(args.scores, allow_pickle=False)
    fusion = json.loads(args.fusion.read_text(encoding="utf-8"))
    details = fusion["details"]
    report = {
        "scope": "diagnostic transfer audit; paired deltas are not deployable thresholds",
        "fusion_summary": {
            "normal_episodes": fusion["normal_episodes"],
            "normal_episode_false_alarms": fusion["normal_episode_false_alarms"],
            "abnormal_episodes": fusion["abnormal_episodes"],
            "abnormal_episode_detections": fusion["abnormal_episode_detections"],
            "active_step_recall": fusion["active_step_recall"],
            "inactive_step_alarm_rate": fusion["inactive_step_alarm_rate"],
            "pre_onset_alarm_episodes": int(
                sum(
                    row["has_active_fault"]
                    and row["delay_steps"] is not None
                    and row["delay_steps"] < 0
                    for row in details
                )
            ),
        },
        "pairs": [
            audit_pair(data, "libero_object_task0", 0, 1),
            audit_pair(data, "libero_goal_task2", 2, 3),
        ],
    }
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
