#!/usr/bin/env python3
"""Compare source-calibrated temporal rules on an opened target diagnostic set.

This is a diagnostic/model-selection utility, not a final holdout evaluator.  Each
rule is reduced to a causal window statistic and calibrated independently from
normal source episodes at an episode-level conformal error rate.
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


RULES = [
    Rule("vote", 1, 1), Rule("vote", 2, 3), Rule("vote", 3, 5),
    Rule("vote", 4, 7), Rule("vote", 5, 9),
    Rule("mean", 0, 3), Rule("mean", 0, 5), Rule("mean", 0, 10),
    Rule("mean", 0, 15),
]


def window_statistic(scores: np.ndarray, rule: Rule) -> np.ndarray:
    """Return a causal statistic ending at every full window."""
    if len(scores) < rule.window:
        return np.empty(0, dtype=float)
    windows = np.lib.stride_tricks.sliding_window_view(scores, rule.window)
    if rule.kind == "mean":
        return windows.mean(axis=1)
    # A k-of-m alarm at step threshold tau is equivalent to the kth largest
    # score in the window exceeding tau.
    return np.sort(windows, axis=1)[:, -rule.k]


def conformal_upper(values: np.ndarray, alpha: float) -> tuple[float, int]:
    """Finite-sample upper conformal quantile (strict > threshold alarms)."""
    values = np.sort(np.asarray(values, dtype=float))
    n = len(values)
    rank = min(n, math.ceil((n + 1) * (1.0 - alpha)))
    return float(values[rank - 1]), rank


def source_episodes(d: np.lib.npyio.NpzFile, beta: float) -> list[np.ndarray]:
    robust = d["abnormal_logp"] - d["normal_logp"] - beta * d["ensemble_std"]
    out = []
    normal = ~d["abnormal"].astype(bool)
    # Source tune file contains one condition per scale; only truly normal
    # scale==1 episodes are admitted to calibration.
    for ep in sorted(np.unique(d["episode"][normal & np.isclose(d["scale"], 1.0)])):
        mask = normal & np.isclose(d["scale"], 1.0) & (d["episode"] == ep)
        idx = np.flatnonzero(mask)[np.argsort(d["action"][mask])]
        out.append(robust[idx])
    return out


def target_episodes(d: np.lib.npyio.NpzFile, beta: float):
    robust = d["abnormal_logp"] - d["normal_logp"] - beta * d["ensemble_std"]
    for fi, ep in sorted(set(zip(d["file_index"].tolist(), d["episode_idx"].tolist()))):
        mask = (d["file_index"] == fi) & (d["episode_idx"] == ep)
        idx = np.flatnonzero(mask)[np.argsort(d["action_index"][mask])]
        yield int(fi), int(ep), robust[idx], d["action_index"][idx], d["abnormal"][idx].astype(bool)


def runs(mask: np.ndarray) -> list[int]:
    lengths, current = [], 0
    for value in mask:
        if value:
            current += 1
        elif current:
            lengths.append(current); current = 0
    if current:
        lengths.append(current)
    return lengths


def evaluate(rule: Rule, threshold: float, episodes) -> dict:
    details = []
    normal_runs = []
    for fi, ep, scores, actions, active in episodes:
        stat = window_statistic(scores, rule)
        end_actions = actions[rule.window - 1:]
        alarm = stat > threshold
        alarm_actions = end_actions[alarm]
        active_hits = alarm_actions[np.isin(alarm_actions, actions[active])]
        inactive_hits = alarm_actions[~np.isin(alarm_actions, actions[active])]
        if not active.any():
            normal_runs.extend(runs(scores > threshold))
        details.append({
            "file_index": fi, "episode_idx": ep, "normal": bool(not active.any()),
            "any_alarm": bool(alarm.any()), "active_detection": bool(len(active_hits)),
            "inactive_alarm": bool(len(inactive_hits)),
            "delay": int(active_hits.min() - actions[active].min()) if len(active_hits) else None,
        })
    normal = [x for x in details if x["normal"]]
    abnormal = [x for x in details if not x["normal"]]
    delays = [x["delay"] for x in abnormal if x["delay"] is not None]
    return {
        "rule": rule.name, "threshold": threshold,
        "normal_episodes": len(normal), "abnormal_episodes": len(abnormal),
        "normal_episode_false_alarm": float(np.mean([x["any_alarm"] for x in normal])),
        "abnormal_episode_active_detection": float(np.mean([x["active_detection"] for x in abnormal])),
        "abnormal_episode_inactive_alarm": float(np.mean([x["inactive_alarm"] for x in abnormal])),
        "median_detection_delay": float(np.median(delays)) if delays else None,
        "normal_raw_score_run_lengths": {
            "count": len(normal_runs),
            "max": max(normal_runs, default=0),
            "median": float(np.median(normal_runs)) if normal_runs else 0.0,
            "ge2": int(sum(x >= 2 for x in normal_runs)),
            "ge3": int(sum(x >= 3 for x in normal_runs)),
            "ge5": int(sum(x >= 5 for x in normal_runs)),
        },
        "details": details,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--target", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--alpha", type=float, default=0.10)
    p.add_argument("--beta", type=float, default=1.0)
    args = p.parse_args()
    source = np.load(args.source)
    target = np.load(args.target)
    source_eps = source_episodes(source, args.beta)
    target_eps = list(target_episodes(target, args.beta))
    results = []
    for rule in RULES:
        episode_stats = np.asarray([
            np.max(window_statistic(scores, rule)) for scores in source_eps
            if len(scores) >= rule.window
        ])
        threshold, rank = conformal_upper(episode_stats, args.alpha)
        result = evaluate(rule, threshold, target_eps)
        result["source_calibration"] = {
            "episodes": len(episode_stats), "alpha": args.alpha,
            "rank": rank, "episode_stat_min": float(episode_stats.min()),
            "episode_stat_median": float(np.median(episode_stats)),
            "episode_stat_max": float(episode_stats.max()),
            "calibration_exceedances": int(np.sum(episode_stats > threshold)),
        }
        results.append(result)
    payload = {
        "diagnostic_only": True,
        "warning": "Opened LIBERO-10 data; do not report as final holdout.",
        "score": "abnormal_logp-normal_logp-beta*ensemble_std",
        "beta": args.beta,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"diagnostic_only": True, "results": [
        {k: v for k, v in r.items() if k not in {"details", "normal_raw_score_run_lengths"}}
        for r in results
    ]}, indent=2))


if __name__ == "__main__":
    main()
