"""Diagnose task-wise overlap of the exploratory visual density experts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from visual_dual_density import load_model, score, score_raw


def auc(labels: np.ndarray, values: np.ndarray) -> float | None:
    pos = values[labels]
    neg = values[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return None
    # Pairwise form handles the small diagnostic sets and ties explicitly.
    return float(((pos[:, None] > neg[None, :]).mean() +
                  0.5 * (pos[:, None] == neg[None, :]).mean()))


def quantiles(values: np.ndarray) -> dict[str, float] | None:
    if len(values) == 0:
        return None
    return {f"q{int(q * 100):02d}": float(np.quantile(values, q))
            for q in (0.0, 0.1, 0.5, 0.9, 1.0)}


def summarize(data: np.lib.npyio.NpzFile, values: np.ndarray) -> dict:
    eligible = data["previous_ambiguous_any"].astype(bool)
    active = data["previous_active_any"].astype(bool)
    clean = np.isclose(data["scale"], 1.0)
    rows = []
    for task in range(10):
        mask = eligible & (data["task_id"] == task) & (clean | active)
        normal = values[mask & clean]
        abnormal = values[mask & active]
        rows.append({
            "task_id": task,
            "normal_n": int(len(normal)),
            "abnormal_n": int(len(abnormal)),
            "normal": quantiles(normal),
            "abnormal": quantiles(abnormal),
            "auc": auc(active[mask], values[mask]),
            "abnormal_above_normal_max_fraction": (
                float(np.mean(abnormal > normal.max())) if len(normal) and len(abnormal) else None
            ),
        })
    mask = eligible & (clean | active)
    return {
        "overall_auc": auc(active[mask], values[mask]),
        "normal": quantiles(values[eligible & clean]),
        "abnormal": quantiles(values[eligible & active]),
        "by_task": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", action="append", required=True,
                        help="NAME=semantic_chunks.npz")
    parser.add_argument("--features", action="append", required=True,
                        help="NAME=head_features.npz")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    data_paths = dict(value.split("=", 1) for value in args.data)
    feature_paths = dict(value.split("=", 1) for value in args.features)
    if set(data_paths) != set(feature_paths):
        raise ValueError("data and feature split names differ")
    model = load_model(args.model)
    report = {}
    for name in data_paths:
        data = np.load(data_paths[name], allow_pickle=False)
        features = np.load(feature_paths[name], allow_pickle=False)["features"]
        report[name] = {
            "standardized_likelihood_ratio": summarize(data, score(model, data, features)),
            "raw_log_likelihood_ratio": summarize(data, score_raw(model, data, features)),
        }
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
