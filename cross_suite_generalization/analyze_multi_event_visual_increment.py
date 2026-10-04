"""Grouped diagnostics for multi-event visual consequence features.

This is development analysis, not a sealed-test result. Entire (suite, task) groups
are held out together so windows from one task cannot cross the fold boundary.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


def auc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    positive = scores[labels == 1]
    negative = scores[labels == 0]
    if not len(positive) or not len(negative):
        return None
    return float(sum((a > b) + 0.5 * (a == b) for a in positive for b in negative)
                 / (len(positive) * len(negative)))


def fit(x: np.ndarray, y: np.ndarray) -> dict:
    mean = x.mean(axis=0)
    scale = np.maximum(x.std(axis=0), 1e-6)
    z = (x - mean) / scale
    counts = np.bincount(y, minlength=2)
    if np.any(counts == 0):
        raise ValueError("training fold requires both outcomes")
    weights = np.asarray([len(y) / (2 * counts[label]) for label in y])

    def objective(parameters):
        logits = parameters[0] + z @ parameters[1:]
        loss = np.logaddexp(0, logits) - y * logits
        return float(np.sum(weights * loss) / np.sum(weights)
                     + 0.5 * np.sum(parameters[1:] ** 2) / len(y))

    result = minimize(objective, np.zeros(x.shape[1] + 1), method="BFGS")
    if not result.success:
        raise RuntimeError(result.message)
    training_scores = expit(result.x[0] + z @ result.x[1:])
    return {
        "mean": mean, "scale": scale, "parameters": result.x,
        "threshold": float(np.percentile(training_scores[y == 0], 95)),
    }


def predict(model: dict, x: np.ndarray) -> np.ndarray:
    z = (x - model["mean"]) / model["scale"]
    return expit(model["parameters"][0] + z @ model["parameters"][1:])


def episode_rows(manifest: dict, tracks: dict) -> list[dict]:
    measured = {(row["episode_id"], row["role"]): row for row in tracks["records"]
                if row["status"] == "measured"}
    rows = []
    for episode in manifest["episodes"]:
        event_values, control_values = [], []
        for index in range(len(episode["label_blind_low_response_events"])):
            event = measured.get((episode["episode_id"], f"event_{index}"))
            control = measured.get((episode["episode_id"], f"control_{index}"))
            if event is None or control is None:
                continue
            event_values.append(float(event["window_median_set_max_residual_px"]))
            control_values.append(float(control["window_median_set_max_residual_px"]))
        if not event_values:
            continue
        responses = np.asarray([
            item["response_ratio"] for item in episode["label_blind_low_response_events"]
        ], dtype=float)
        event_log = np.log(np.asarray(event_values) + 1e-4)
        control_log = np.log(np.asarray(control_values) + 1e-4)
        rows.append({
            "episode_id": episode["episode_id"],
            "group": f'{episode["suite"]}:task{episode["task_id"]}',
            "suite": episode["suite"],
            "failure": not bool(episode["success"]),
            "steps": int(episode["steps"]),
            "mean_log_response": float(np.mean(np.log(responses + 1e-4))),
            "min_log_response": float(np.min(np.log(responses + 1e-4))),
            "mean_log_event_visual": float(np.mean(event_log)),
            "mean_visual_log_drop": float(np.mean(control_log - event_log)),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--tracks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    tracks = json.loads(args.tracks.read_text(encoding="utf-8"))
    rows = episode_rows(manifest, tracks)
    labels = np.asarray([row["failure"] for row in rows], dtype=int)
    groups = np.asarray([row["group"] for row in rows])
    specs = {
        "action_only": ["mean_log_response", "min_log_response"],
        "visual_only": ["mean_log_event_visual", "mean_visual_log_drop"],
        "action_visual_fusion": [
            "mean_log_response", "min_log_response",
            "mean_log_event_visual", "mean_visual_log_drop",
        ],
    }
    metrics = {}
    for name, fields in specs.items():
        x = np.asarray([[row[field] for field in fields] for row in rows], dtype=float)
        scores = np.full(len(rows), np.nan)
        alarms = np.zeros(len(rows), dtype=bool)
        fold_details = []
        for held_out in sorted(set(groups)):
            test = groups == held_out
            train = ~test
            model = fit(x[train], labels[train])
            fold_scores = predict(model, x[test])
            scores[test] = fold_scores
            alarms[test] = fold_scores >= model["threshold"]
            fold_details.append({
                "held_out_group": held_out,
                "episodes": int(test.sum()),
                "failures": int(labels[test].sum()),
                "alarms": int(alarms[test].sum()),
            })
        metrics[name] = {
            "features": fields,
            "grouped_out_of_fold_auc": auc(scores, labels),
            "grouped_out_of_fold_failure_detection": float(np.mean(alarms[labels == 1])),
            "grouped_out_of_fold_success_alarm_rate": float(np.mean(alarms[labels == 0])),
            "folds": fold_details,
        }
    suite_counts = {}
    for suite in sorted({row["suite"] for row in rows}):
        selected = [row for row in rows if row["suite"] == suite]
        suite_counts[suite] = {
            "episodes": len(selected),
            "failures": sum(row["failure"] for row in selected),
        }
    payload = {
        "schema_version": 1,
        "status": "development_grouped_diagnostic_not_sealed_test",
        "episode_count": len(rows),
        "failure_count": int(labels.sum()),
        "success_count": int((1 - labels).sum()),
        "group_count": len(set(groups)),
        "suite_counts": suite_counts,
        "outcome_by_group": dict(Counter(
            f'{row["group"]}|{"failure" if row["failure"] else "success"}' for row in rows
        )),
        "metrics": metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "episodes": len(rows), "failures": int(labels.sum()),
        "metrics": {name: {
            "auc": value["grouped_out_of_fold_auc"],
            "detection": value["grouped_out_of_fold_failure_detection"],
            "success_alarm": value["grouped_out_of_fold_success_alarm_rate"],
        } for name, value in metrics.items()},
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
