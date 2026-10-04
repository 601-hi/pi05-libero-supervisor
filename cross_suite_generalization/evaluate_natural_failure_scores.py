#!/usr/bin/env python3
"""Evaluate frozen per-step scores against natural episode outcomes.

The scored NPZ and ``--trace`` arguments must use the same file order.  Thresholds
come only from a separate source-normal calibration NPZ; target outcomes are used
for reporting, never threshold selection.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Rule:
    kind: str
    k: int
    window: int

    @property
    def name(self) -> str:
        return f"{self.k}of{self.window}" if self.kind == "vote" else f"mean{self.window}"


RULES = (
    Rule("vote", 1, 1), Rule("vote", 2, 3), Rule("vote", 3, 5),
    Rule("vote", 4, 7), Rule("vote", 5, 9),
    Rule("mean", 0, 3), Rule("mean", 0, 5), Rule("mean", 0, 10),
)


def window_statistic(scores: np.ndarray, rule: Rule) -> np.ndarray:
    if len(scores) < rule.window:
        return np.empty(0, dtype=float)
    windows = np.lib.stride_tricks.sliding_window_view(scores, rule.window)
    if rule.kind == "mean":
        return windows.mean(axis=1)
    return np.sort(windows, axis=1)[:, -rule.k]


def conformal_upper(values: np.ndarray, alpha: float) -> tuple[float, int]:
    values = np.sort(np.asarray(values, dtype=float))
    if not len(values):
        raise ValueError("empty source calibration")
    rank = min(len(values), math.ceil((len(values) + 1) * (1.0 - alpha)))
    return float(values[rank - 1]), rank


def load_outcomes(paths: list[Path]) -> dict[tuple[int, int], dict]:
    outcomes = {}
    for file_index, path in enumerate(paths):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if row.get("event") == "episode_end":
                    key = (file_index, int(row["episode_idx"]))
                    if key in outcomes:
                        raise ValueError(f"duplicate episode_end for {key} in {path}")
                    outcomes[key] = {
                        "success": bool(row.get("success", False)),
                        "task_id": int(row["task_id"]),
                        "trace": str(path),
                    }
    return outcomes


def source_episodes(data: np.lib.npyio.NpzFile, beta: float) -> list[np.ndarray]:
    score = data["abnormal_logp"] - data["normal_logp"] - beta * data["ensemble_std"]
    normal = ~data["abnormal"].astype(bool)
    if "scale" in data.files:
        normal &= np.isclose(data["scale"], 1.0)
    episode_key = "episode" if "episode" in data.files else "episode_idx"
    action_key = "action" if "action" in data.files else "action_index"
    episodes = []
    for episode in np.unique(data[episode_key][normal]):
        mask = normal & (data[episode_key] == episode)
        indices = np.flatnonzero(mask)
        indices = indices[np.argsort(data[action_key][indices])]
        episodes.append(score[indices])
    return episodes


def target_episodes(data: np.lib.npyio.NpzFile, outcomes: dict, beta: float):
    score = data["abnormal_logp"] - data["normal_logp"] - beta * data["ensemble_std"]
    for key in sorted(outcomes):
        file_index, episode_idx = key
        mask = (data["file_index"] == file_index) & (data["episode_idx"] == episode_idx)
        indices = np.flatnonzero(mask)
        if not len(indices):
            continue
        indices = indices[np.argsort(data["action_index"][indices])]
        yield key, score[indices], data["action_index"][indices], outcomes[key]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--trace", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--beta", type=float, default=1.0)
    args = parser.parse_args()

    source = np.load(args.source)
    target = np.load(args.target)
    outcomes = load_outcomes(args.trace)
    source_eps = source_episodes(source, args.beta)
    target_eps = list(target_episodes(target, outcomes, args.beta))
    missing = sorted(set(outcomes) - {key for key, *_ in target_eps})
    if missing:
        raise ValueError(f"{len(missing)} outcomes have no score rows: {missing[:5]}")

    results = []
    for rule in RULES:
        calibration = np.asarray([
            np.max(window_statistic(scores, rule))
            for scores in source_eps if len(scores) >= rule.window
        ])
        threshold, rank = conformal_upper(calibration, args.alpha)
        details = []
        for (file_index, episode_idx), scores, actions, outcome in target_eps:
            statistic = window_statistic(scores, rule)
            alarm_mask = statistic > threshold
            first = None
            if np.any(alarm_mask):
                first_window = int(np.flatnonzero(alarm_mask)[0])
                first = int(actions[first_window + rule.window - 1])
            details.append({
                "file_index": file_index,
                "episode_idx": episode_idx,
                **outcome,
                "alarm": first is not None,
                "first_alarm_action_index": first,
                "max_statistic": float(np.max(statistic)) if len(statistic) else None,
            })
        successes = [x for x in details if x["success"]]
        failures = [x for x in details if not x["success"]]
        by_file = {}
        for file_index, path in enumerate(args.trace):
            rows = [x for x in details if x["file_index"] == file_index]
            ok = [x for x in rows if x["success"]]
            bad = [x for x in rows if not x["success"]]
            by_file[str(path)] = {
                "episodes": len(rows), "successes": len(ok), "failures": len(bad),
                "success_alarm_rate": float(np.mean([x["alarm"] for x in ok])) if ok else None,
                "failure_detection_rate": float(np.mean([x["alarm"] for x in bad])) if bad else None,
            }
        results.append({
            "rule": rule.name,
            "threshold": threshold,
            "source_calibration_episodes": len(calibration),
            "source_conformal_rank": rank,
            "successful_episodes": len(successes),
            "failed_episodes": len(failures),
            "successful_episode_alarm_rate": (
                float(np.mean([x["alarm"] for x in successes])) if successes else None
            ),
            "failed_episode_detection_rate": (
                float(np.mean([x["alarm"] for x in failures])) if failures else None
            ),
            "by_file": by_file,
            "details": details,
        })

    payload = {
        "evaluation_type": "frozen_zero_fit_natural_outcome_diagnostic",
        "target_used_for_threshold_selection": False,
        "positive_label": "episode_end.success == false",
        "score": "abnormal_logp-normal_logp-beta*ensemble_std",
        "alpha": args.alpha,
        "beta": args.beta,
        "traces": [str(x) for x in args.trace],
        "outcomes": len(outcomes),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    compact = [{k: v for k, v in row.items() if k != "details"} for row in results]
    print(json.dumps({"outcomes": len(outcomes), "results": compact}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
