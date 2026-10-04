#!/usr/bin/env python3
"""Fair replay of the frozen legacy v2 binary MLP on seed21 traces.

The legacy model is reconstructed deterministically by its original frozen
training function. Seed21 labels are used only to score already-produced
causal alarms, never as model inputs or for threshold selection.
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np

import evaluate_supervisor as ev
import evaluate_v2_seed14_frozen as frozen
import train_hybrid_supervisor_v2 as v2


def evaluate_file(path: Path, model, cov, predict, thresholds, source_id: int, planned_scale: float):
    rows, (rec, _, x, y, _) = v2.load(str(path))
    features = v2.physical_features(model, cov, rec, x, y)
    probabilities = predict(features)
    normalized = np.asarray([
        p / (thresholds[r["task_id"]] + 1e-12) for p, r in zip(probabilities, rec)
    ])
    alarms = ev.persistence_series(normalized, rec, 1.0, required=2, window=3)
    active = v2.active_flags(rows, rec)
    groups = ev.episode_groups(rec)
    details = []
    for (task_id, episode_idx), indices in groups.items():
        indices = sorted(indices, key=lambda i: int(rec[i]["action_index"]))
        mask = active[indices]
        active_indices = [i for i in indices if active[i]]
        alarm_indices = [i for i in indices if alarms[i]]
        detected = any(alarms[i] and active[i] for i in indices)
        first_active_local = next((j for j, i in enumerate(indices) if active[i]), None)
        first_alarm_local = next((j for j, i in enumerate(indices) if alarms[i]), None)
        focus = active_indices if active_indices else indices
        details.append({
            "episode_id": f"source{source_id}:task{task_id}:episode{episode_idx}",
            "task_id": int(task_id),
            "episode_idx": int(episode_idx),
            "planned_scale": float(planned_scale),
            "steps": len(indices),
            "active_steps": int(mask.sum()),
            "detected_during_active": bool(detected),
            "any_alarm": bool(alarm_indices),
            "alarm_steps": [int(rec[i]["action_index"]) for i in alarm_indices],
            "delay_from_active": None if first_active_local is None or first_alarm_local is None else int(first_alarm_local - first_active_local),
            "max_probability_focus": float(max(probabilities[i] for i in focus)),
            "max_normalized_risk_focus": float(max(normalized[i] for i in focus)),
            "mean_target_norm_focus": float(np.mean([rec[i]["target_norm"] for i in focus])),
        })
    return details, alarms, active


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--normal", type=Path, required=True)
    p.add_argument("--scale025", type=Path, required=True)
    p.add_argument("--scale050", type=Path, required=True)
    p.add_argument("--scale075", type=Path, required=True)
    p.add_argument("--reference-report", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--allow-reconstructed", action="store_true",
                   help="Evaluate a deterministic legacy-architecture reconstruction when exact frozen weights were not saved")
    args = p.parse_args()

    model, cov, predict, thresholds, loss = frozen.fit_frozen()
    reference = json.loads(args.reference_report.read_text(encoding="utf-8"))
    reference_thresholds = {int(k): float(v) for k, v in reference["frozen_rule"]["thresholds"].items()}
    max_threshold_difference = max(abs(thresholds[k] - reference_thresholds[k]) for k in thresholds)
    if max_threshold_difference > 1e-12 and not args.allow_reconstructed:
        raise RuntimeError(f"legacy v2 reconstruction mismatch: {max_threshold_difference}")

    all_details = []
    step_active, step_alarm = [], []
    paths = ((args.normal, 1.0), (args.scale025, 0.25), (args.scale050, 0.50), (args.scale075, 0.75))
    for source_id, (path, planned_scale) in enumerate(paths):
        details, alarms, active = evaluate_file(path, model, cov, predict, thresholds, source_id, planned_scale)
        all_details.extend(details)
        step_alarm.append(alarms)
        step_active.append(active)
    alarms = np.concatenate(step_alarm)
    active = np.concatenate(step_active)
    clean = [d for d in all_details if d["active_steps"] == 0]
    faults = [d for d in all_details if d["active_steps"] > 0]
    by_scale = {}
    for scale in sorted({d["planned_scale"] for d in all_details}):
        subset = [d for d in all_details if np.isclose(d["planned_scale"], scale)]
        exposed = [d for d in subset if d["active_steps"] > 0]
        by_scale[str(scale)] = {
            "planned_episodes": len(subset),
            "exposed_episodes": len(exposed),
            "zero_exposure_episodes": len(subset) - len(exposed),
            "detected_during_active": sum(d["detected_during_active"] for d in exposed),
            "any_alarm": sum(d["any_alarm"] for d in subset),
        }
    report = {
        "integrity": {
            "model": "legacy_v2_binary_mlp_reconstructed_architecture",
            "exact_frozen_weights_available": False,
            "reconstruction_matches_seed14_frozen_thresholds": max_threshold_difference <= 1e-12,
            "max_threshold_difference": max_threshold_difference,
            "evidence_level": "architecture-and-data-matched reconstruction; not an exact frozen-checkpoint replay",
            "train_loss": loss,
            "persistence": "2of3",
            "seed21_labels_used_for_training_or_tuning": False,
        },
        "episodes": len(all_details),
        "normal_episodes": len(clean),
        "normal_episode_false_alarms": sum(d["any_alarm"] for d in clean),
        "abnormal_episodes": len(faults),
        "abnormal_episode_detections": sum(d["detected_during_active"] for d in faults),
        "active_step_recall": float(alarms[active].mean()),
        "inactive_step_alarm_rate": float(alarms[~active].mean()),
        "by_scale": by_scale,
        "false_positives": [d for d in clean if d["any_alarm"]],
        "false_negatives": [d for d in faults if not d["detected_during_active"]],
        "details": all_details,
    }
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in {"details", "false_positives", "false_negatives"}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
