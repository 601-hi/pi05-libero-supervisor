#!/usr/bin/env python3
"""Cross-task diagnostic for normal-only context-adaptive score thresholds."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from compare_temporal_decision_rules import RULES, Rule, window_statistic
from conditional_relational_dual_expert import conditional_features
from libero_droid_transfer import load_trace
from conditional_relational_dual_expert import HIGH, LOW


def trace_path(trace_dir: Path, task: int, file_index: int) -> Path:
    suffix = {0: "normal", 1: "scale025", 2: "scale050", 3: "scale075"}[file_index % 4]
    return trace_dir / f"libero_10_task{task}_seed32_{suffix}.jsonl"


def rebuild_context(scores, trace_dir: Path) -> np.ndarray:
    context = np.empty((len(scores["normal_logp"]), 21), np.float32)
    for fi in sorted(np.unique(scores["file_index"])):
        mask = scores["file_index"] == fi
        task_values = np.unique(scores["task_id"][mask])
        if len(task_values) != 1:
            raise ValueError(f"file {fi} has multiple tasks")
        path = trace_path(trace_dir, int(task_values[0]), int(fi))
        data, meta = load_trace(path)
        episode = np.asarray(meta["episode_idx"], np.int32)
        x, _ = conditional_features(data, episode, 20, gripper_scale=1.0)
        target_idx = np.flatnonzero(mask)
        order = np.lexsort((scores["action_index"][mask], scores["episode_idx"][mask]))
        target_idx = target_idx[order]
        source_order = np.lexsort((meta["action_index"], meta["episode_idx"]))
        for key in ("episode_idx", "action_index", "task_id", "abnormal"):
            if not np.array_equal(np.asarray(meta[key])[source_order], scores[key][target_idx]):
                raise ValueError(f"alignment failure for file={fi}, key={key}")
        context[target_idx] = x[source_order]
    return context


def rebuild_mechanism_regime(scores, trace_dir: Path) -> np.ndarray:
    """Exploratory, pre-action-only mechanism bins for the opened diagnostic set."""
    regime = np.empty(len(scores["normal_logp"]), np.int8)
    for fi in sorted(np.unique(scores["file_index"])):
        mask = scores["file_index"] == fi
        task = int(np.unique(scores["task_id"][mask])[0])
        rows = []
        with trace_path(trace_dir, task, int(fi)).open("r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if row.get("event") == "step" and int(row["action_index"]) >= 5:
                    rows.append(row)
        rows.sort(key=lambda r: (int(r["episode_idx"]), int(r["action_index"])))
        target_idx = np.flatnonzero(mask)
        order = np.lexsort((scores["action_index"][mask], scores["episode_idx"][mask]))
        target_idx = target_idx[order]
        if len(rows) != len(target_idx):
            raise ValueError(f"mechanism alignment length failure for file={fi}")
        for dst, row in zip(target_idx, rows):
            q = np.asarray(row["joint_pos_before"], dtype=float)
            margin = np.min(np.minimum((q - LOW) / (HIGH - LOW), (HIGH - q) / (HIGH - LOW)))
            large = np.linalg.norm(row["intended_target_translation"]) >= 0.035
            slow = np.linalg.norm(row["joint_vel_before"]) < 0.25
            moving_gripper = np.linalg.norm(row["gripper_qvel_before"]) > 0.005
            special = margin < 0.10 or (large and slow) or (large and moving_gripper)
            if special:
                regime[dst] = 0
            elif large:
                regime[dst] = 1
            elif slow:
                regime[dst] = 2
            else:
                regime[dst] = 3
    return regime


def mechanism_standardize(raw, regime, fit):
    result = np.empty_like(raw, dtype=float)
    global_center = float(np.median(raw[fit]))
    global_scale = float(np.percentile(raw[fit], 75) - np.percentile(raw[fit], 25))
    global_scale = max(global_scale, 1e-3)
    for group in range(4):
        selected = fit[regime[fit] == group]
        if len(selected) < 20:
            center, scale = global_center, global_scale
        else:
            center = float(np.median(raw[selected]))
            scale = float(np.percentile(raw[selected], 75) - np.percentile(raw[selected], 25))
            scale = max(scale, 1e-3)
        mask = regime == group
        result[mask] = (raw[mask] - center) / scale
    return result


def design_fit(x: np.ndarray):
    center = np.median(x, axis=0)
    scale = np.percentile(x, 75, axis=0) - np.percentile(x, 25, axis=0)
    scale = np.where(scale > 1e-6, scale, 1.0)
    z = np.clip((x - center) / scale, -8.0, 8.0)
    return np.c_[np.ones(len(z)), z, z * z], center, scale


def design_apply(x: np.ndarray, center: np.ndarray, scale: np.ndarray):
    z = np.clip((x - center) / scale, -8.0, 8.0)
    return np.c_[np.ones(len(z)), z, z * z]


def ridge_model(train_x, train_y, ridge: float):
    a, center, scale = design_fit(train_x)
    penalty = np.eye(a.shape[1]); penalty[0, 0] = 0.0
    coef = np.linalg.solve(a.T @ a + ridge * penalty, a.T @ train_y)
    return coef, center, scale


def ridge_apply(model, test_x):
    coef, center, scale = model
    return design_apply(test_x, center, scale) @ coef


def episode_indices(d, normal_files_only=False):
    allowed = np.ones(len(d["file_index"]), bool)
    if normal_files_only:
        allowed = d["file_index"] % 4 == 0
    for fi, ep in sorted(set(zip(d["file_index"][allowed], d["episode_idx"][allowed]))):
        mask = allowed & (d["file_index"] == fi) & (d["episode_idx"] == ep)
        idx = np.flatnonzero(mask)
        idx = idx[np.argsort(d["action_index"][idx])]
        yield int(fi), int(ep), idx


def max_stat(score, idx, rule):
    values = window_statistic(score[idx], rule)
    return float(values.max())


def detect(score, d, idx, rule, threshold):
    stat = window_statistic(score[idx], rule)
    actions = d["action_index"][idx]
    active = d["abnormal"][idx].astype(bool)
    ends = actions[rule.window - 1:]
    hits = ends[(stat > threshold) & np.isin(ends, actions[active])]
    return bool(len(hits)), (int(hits.min() - actions[active].min()) if len(hits) else None)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scores", type=Path, required=True)
    p.add_argument("--trace-directory", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--ridge", type=float, default=10.0)
    args = p.parse_args()
    d = np.load(args.scores)
    raw = d["abnormal_logp"] - d["normal_logp"] - d["ensemble_std"]
    x = rebuild_context(d, args.trace_directory)
    regime = rebuild_mechanism_regime(d, args.trace_directory)
    normal_eps = list(episode_indices(d, normal_files_only=True))
    tasks = sorted(set(int(d["task_id"][idx[0]]) for _, _, idx in normal_eps))
    selected = [r for r in RULES if r.name in {"1of1", "2of3", "3of5", "4of7", "5of9", "mean10"}]
    output = []
    # Two folds: one episode per non-held-out task fits the normal-score baseline;
    # the other episode calibrates a max-stat threshold. Both episodes of the
    # held-out task test normal false alarms. No abnormal label enters fitting.
    for fit_ep in (0, 1):
        for held_task in tasks:
            fit = np.concatenate([idx for _, ep, idx in normal_eps
                                  if ep == fit_ep and int(d["task_id"][idx[0]]) != held_task])
            cal_eps = [idx for _, ep, idx in normal_eps
                       if ep != fit_ep and int(d["task_id"][idx[0]]) != held_task]
            test_normal = [idx for _, _, idx in normal_eps if int(d["task_id"][idx[0]]) == held_task]
            mean_model = ridge_model(x[fit], raw[fit], args.ridge)
            prediction = ridge_apply(mean_model, x)
            residual = raw - prediction
            fit_residual = raw[fit] - ridge_apply(mean_model, x[fit])
            residual_floor = max(float(np.median(np.abs(fit_residual))) * 0.1, 1e-3)
            scale_model = ridge_model(
                x[fit], np.log(np.abs(fit_residual) + residual_floor), args.ridge
            )
            predicted_scale = np.exp(np.clip(ridge_apply(scale_model, x), -5.0, 5.0))
            standardized_residual = residual / predicted_scale
            mechanism_score = mechanism_standardize(raw, regime, fit)
            abnormal_eps = [(fi, idx) for fi, _, idx in episode_indices(d)
                            if fi % 4 != 0 and int(d["task_id"][idx[0]]) == held_task]
            for score_name, score in (
                ("raw", raw),
                ("context_residual", residual),
                ("context_standardized_residual", standardized_residual),
                ("mechanism_standardized", mechanism_score),
            ):
                for rule in selected:
                    threshold = max(max_stat(score, idx, rule) for idx in cal_eps)
                    false_alarms = [max_stat(score, idx, rule) > threshold for idx in test_normal]
                    detections, delays, detection_details = [], [], []
                    for fi, idx in abnormal_eps:
                        hit, delay = detect(score, d, idx, rule, threshold)
                        detections.append(hit)
                        if delay is not None: delays.append(delay)
                        detection_details.append({
                            "scale": {1: 0.25, 2: 0.50, 3: 0.75}[fi % 4],
                            "detected": hit, "delay": delay,
                        })
                    output.append({
                        "fit_episode": fit_ep, "held_out_task": held_task,
                        "score": score_name, "rule": rule.name, "threshold": threshold,
                        "normal_test_episodes": len(false_alarms),
                        "normal_false_alarms": int(sum(false_alarms)),
                        "abnormal_test_episodes": len(detections),
                        "abnormal_active_detections": int(sum(detections)),
                        "delays": delays,
                        "detection_details": detection_details,
                    })
    summary = []
    for score_name in (
        "raw", "context_residual", "context_standardized_residual", "mechanism_standardized"
    ):
        for rule in selected:
            rows = [r for r in output if r["score"] == score_name and r["rule"] == rule.name]
            delays = sum((r["delays"] for r in rows), [])
            summary.append({
                "score": score_name, "rule": rule.name,
                "normal_episode_false_alarm": sum(r["normal_false_alarms"] for r in rows) / sum(r["normal_test_episodes"] for r in rows),
                "abnormal_episode_active_detection": sum(r["abnormal_active_detections"] for r in rows) / sum(r["abnormal_test_episodes"] for r in rows),
                "median_delay": float(np.median(delays)) if delays else None,
                "by_scale": {
                    str(scale): {
                        "episodes": sum(1 for r in rows for x in r["detection_details"] if x["scale"] == scale),
                        "detections": sum(int(x["detected"]) for r in rows for x in r["detection_details"] if x["scale"] == scale),
                    }
                    for scale in (0.25, 0.50, 0.75)
                },
            })
    result = {
        "diagnostic_only": True,
        "protocol": "leave-one-task-out, two fit/calibration episode folds",
        "normal_labels_only_for_adaptation": True,
        "mechanism_bins_are_exploratory": True,
        "ridge": args.ridge,
        "summary": summary,
        "folds": output,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
