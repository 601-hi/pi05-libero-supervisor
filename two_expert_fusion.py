#!/usr/bin/env python3
"""Calibration and causal four-state fusion for normal/abnormal log likelihoods."""
from __future__ import annotations

import argparse
import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class FusionResult:
    state: str
    log_likelihood_ratio: float
    persistent_alarm: bool


class TwoExpertFusion:
    def __init__(self, config):
        self.c = config; self.history = deque(maxlen=int(config["persistence_window"]))

    def reset(self): self.history.clear()

    def update(self, normal_logp: float, abnormal_logp: float, task_id: int | None = None) -> FusionResult:
        c = self.c
        if "by_task" in c:
            if task_id is None: raise ValueError("task_id is required for task-conditional calibration")
            c = {**c, **c["by_task"][str(task_id)]}
        ratio = abnormal_logp - normal_logp
        ratio_evidence = (ratio - c.get("ratio_center", 0.0)) / c.get("ratio_scale", 1.0)
        novelty_evidence = (c.get("normal_logp_center", normal_logp) - normal_logp) / c.get("normal_logp_scale", 1.0)
        n_high = normal_logp >= c["normal_absolute_floor"]
        a_high = abnormal_logp >= c["abnormal_absolute_floor"]
        if n_high and a_high:
            state = "known_abnormal" if ratio_evidence >= c["ratio_abnormal_threshold"] else "ambiguous_overlap"
        elif a_high:
            state = "known_abnormal" if ratio_evidence >= c["ratio_abnormal_threshold"] else "ambiguous_overlap"
        elif n_high:
            state = "normal"
        else:
            state = "unknown_abnormal"
        unknown_alarm = state == "unknown_abnormal" and novelty_evidence >= c.get("unknown_novelty_threshold", float("inf"))
        point = state == "known_abnormal" or unknown_alarm
        self.history.append(point)
        persistent = sum(self.history) >= int(self.c["persistence_required"])
        return FusionResult(state, float(ratio), persistent)


def _calibrate_one(normal_logp, abnormal_logp, labels, clean_normal, normal_quantile, abnormal_quantile):
    labels = np.asarray(labels, dtype=bool); clean_normal = np.asarray(clean_normal, dtype=bool)
    nlp = np.asarray(normal_logp); alp = np.asarray(abnormal_logp)
    if not clean_normal.any() or not labels.any(): raise ValueError("calibration requires clean-normal and active-abnormal rows")
    normal_floor = float(np.quantile(nlp[clean_normal], normal_quantile))
    abnormal_floor = float(np.quantile(alp[labels], abnormal_quantile))
    ratio = alp - nlp
    # A conservative threshold: 95th percentile of normal evidence ratios.
    ratio_threshold = float(np.quantile(ratio[clean_normal], 0.95))
    return {"normal_absolute_floor": normal_floor, "abnormal_absolute_floor": abnormal_floor,
            "ratio_abnormal_threshold": ratio_threshold}


def calibrate(normal_logp, abnormal_logp, labels, task_ids=None, clean_normal=None,
              normal_quantile=0.01, abnormal_quantile=0.01):
    nlp, alp, labels = np.asarray(normal_logp), np.asarray(abnormal_logp), np.asarray(labels, dtype=bool)
    clean_normal = ~labels if clean_normal is None else np.asarray(clean_normal, dtype=bool)
    base = {"persistence_required": 2, "persistence_window": 3, "calibration_only": True,
            "interpretation": "absolute likelihood gates plus abnormal-minus-normal log likelihood ratio"}
    if task_ids is None:
        return {**base, **_calibrate_one(nlp, alp, labels, clean_normal, normal_quantile, abnormal_quantile)}
    task_ids = np.asarray(task_ids)
    base["by_task"] = {str(task): _calibrate_one(nlp[task_ids == task], alp[task_ids == task],
                                                   labels[task_ids == task], clean_normal[task_ids == task],
                                                   normal_quantile, abnormal_quantile)
                       for task in sorted(np.unique(task_ids))}
    return base


def main():
    parser = argparse.ArgumentParser(description="Calibrate four-state fusion from a calibration-only NPZ")
    parser.add_argument("--scores", type=Path, required=True,
                        help="NPZ with normal_logp, abnormal_logp, abnormal arrays")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(); data = np.load(args.scores, allow_pickle=False)
    clean_normal = np.isclose(data["scale"], 1.0) if "scale" in data.files else ~data["abnormal"]
    config = calibrate(data["normal_logp"], data["abnormal_logp"], data["abnormal"],
                       data.get("task_id"), clean_normal=clean_normal)
    args.out.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(config, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
