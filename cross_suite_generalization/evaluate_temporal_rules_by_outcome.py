#!/usr/bin/env python3
"""Retrospectively score temporal rules against task outcome, not scale windows.

This is an opened-set diagnostic.  It asks whether an episode ever alarms before
its final outcome.  Successful perturbed episodes remain successful negatives;
the disturbance flag is never used as the class label.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np

from compare_temporal_decision_rules import RULES, conformal_upper, source_episodes, window_statistic


PATTERN = re.compile(r"libero_10_task(?P<task>\d+)_seed32_(normal|scale025|scale050|scale075)")


def load_outcomes(trace_root: Path) -> tuple[list[Path], dict[tuple[int, int], bool]]:
    paths = [p for p in sorted(trace_root.glob("*.jsonl")) if PATTERN.fullmatch(p.stem)]
    outcomes = {}
    for file_index, path in enumerate(paths):
        expected_task = int(PATTERN.fullmatch(path.stem)["task"])
        for line in path.open(encoding="utf-8"):
            row = json.loads(line)
            if row.get("event") == "episode_end":
                assert int(row["task_id"]) == expected_task
                outcomes[(file_index, int(row["episode_idx"]))] = bool(row.get("success", False))
    return paths, outcomes


def target_episodes(d, outcomes, beta):
    score = d["abnormal_logp"] - d["normal_logp"] - beta * d["ensemble_std"]
    for fi, ep in sorted(set(zip(d["file_index"].tolist(), d["episode_idx"].tolist()))):
        mask = (d["file_index"] == fi) & (d["episode_idx"] == ep)
        idx = np.flatnonzero(mask)[np.argsort(d["action_index"][mask])]
        key = (int(fi), int(ep))
        if key in outcomes:
            yield key, score[idx], bool(outcomes[key])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--trace-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--beta", type=float, default=1.0)
    args = parser.parse_args()

    paths, outcomes = load_outcomes(args.trace_root)
    source = np.load(args.source)
    target = np.load(args.target)
    source_eps = source_episodes(source, args.beta)
    target_eps = list(target_episodes(target, outcomes, args.beta))
    results = []
    for rule in RULES:
        calibration = np.asarray([
            np.max(window_statistic(x, rule)) for x in source_eps if len(x) >= rule.window
        ])
        threshold, rank = conformal_upper(calibration, args.alpha)
        details = []
        for (fi, ep), scores, success in target_eps:
            stat = window_statistic(scores, rule)
            first = int(np.flatnonzero(stat > threshold)[0] + rule.window - 1) if np.any(stat > threshold) else None
            details.append({"file_index": fi, "episode_idx": ep, "success": success,
                            "alarm": first is not None, "first_alarm_scored_index": first})
        positives = [x for x in details if not x["success"]]
        negatives = [x for x in details if x["success"]]
        results.append({
            "rule": rule.name, "threshold": threshold,
            "source_calibration_rank": rank,
            "successful_episodes": len(negatives),
            "failed_episodes": len(positives),
            "successful_episode_alarm_rate": float(np.mean([x["alarm"] for x in negatives])) if negatives else None,
            "failed_episode_detection_rate": float(np.mean([x["alarm"] for x in positives])) if positives else None,
            "details": details,
        })
    result = {
        "diagnostic_only": True,
        "opened_target": "LIBERO-10 seed32",
        "positive_label": "episode_end.success == false",
        "intervention_used_as_label": False,
        "important_limit": "Very few target failures; rates have high uncertainty and are not a final benchmark.",
        "trace_files": len(paths), "outcomes": len(outcomes), "scored_episodes": len(target_eps),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({**{k: result[k] for k in ("trace_files", "outcomes", "scored_episodes")},
                      "results": [{k: v for k, v in x.items() if k != "details"} for x in results]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
