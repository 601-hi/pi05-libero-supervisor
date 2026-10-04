"""Diagnose feature and decision shift between development and model-selection seeds."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from analyze_multi_event_visual_increment import episode_rows


FEATURES = [
    "mean_log_response", "min_log_response",
    "mean_log_event_visual", "mean_visual_log_drop",
]


def summarize(rows: list[dict]) -> dict:
    result = {}
    for feature in FEATURES:
        values = np.asarray([row[feature] for row in rows], dtype=float)
        result[feature] = {
            "mean": float(values.mean()), "median": float(np.median(values)),
            "std": float(values.std()),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--development-manifest", type=Path, required=True)
    parser.add_argument("--development-tracks", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--selection-tracks", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    def load_rows(manifest_path, track_path):
        return episode_rows(
            json.loads(manifest_path.read_text(encoding="utf-8")),
            json.loads(track_path.read_text(encoding="utf-8")),
        )
    development = load_rows(args.development_manifest, args.development_tracks)
    selection = load_rows(args.selection_manifest, args.selection_tracks)
    scored = json.loads(args.scores.read_text(encoding="utf-8"))["metrics"]
    score_maps = {
        name: {row["episode_id"]: row for row in metric["scores"]}
        for name, metric in scored.items()
    }

    by_partition = {}
    for name, rows in (("development", development), ("model_selection", selection)):
        by_partition[name] = {
            "all": summarize(rows),
            "success": summarize([row for row in rows if not row["failure"]]),
            "failure": summarize([row for row in rows if row["failure"]]),
        }
    dev_success = [row for row in development if not row["failure"]]
    shifts = {}
    for feature in FEATURES:
        dev = np.asarray([row[feature] for row in dev_success])
        sel = np.asarray([row[feature] for row in selection if not row["failure"]])
        shifts[feature] = {
            "success_mean_shift_in_dev_success_sd": float((sel.mean() - dev.mean()) / max(dev.std(), 1e-6)),
            "development_success_median": float(np.median(dev)),
            "selection_success_median": float(np.median(sel)),
        }

    cases = []
    for row in selection:
        action = score_maps["action_only"][row["episode_id"]]
        visual = score_maps["visual_only"][row["episode_id"]]
        fusion = score_maps["action_visual_fusion"][row["episode_id"]]
        if row["failure"] or action["alarm"] or visual["alarm"] or fusion["alarm"]:
            cases.append({
                "episode_id": row["episode_id"], "group": row["group"],
                "failure": row["failure"],
                "action_alarm": action["alarm"], "visual_alarm": visual["alarm"],
                "fusion_alarm": fusion["alarm"],
                **{feature: row[feature] for feature in FEATURES},
            })
    failure_groups = defaultdict(lambda: {"episodes": 0, "failures": 0})
    for row in selection:
        failure_groups[row["group"]]["episodes"] += 1
        failure_groups[row["group"]]["failures"] += int(row["failure"])
    payload = {
        "schema_version": 1,
        "partitions": by_partition,
        "normal_feature_shift": shifts,
        "model_selection_groups_with_failures": {
            key: value for key, value in sorted(failure_groups.items()) if value["failures"]
        },
        "alarm_or_failure_cases": cases,
        "interpretation_guard": (
            "This diagnostic may guide one model-selection decision. Seed38 remains sealed and "
            "must not be used to revise features or thresholds."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "normal_feature_shift": shifts,
        "failure_groups": payload["model_selection_groups_with_failures"],
        "cases": len(cases),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
