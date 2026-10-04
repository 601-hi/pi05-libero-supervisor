#!/usr/bin/env python3
"""Normal-only calibration of causal rolling innovation scores for subtle persistent faults."""
import json
import pathlib

import numpy as np
import torch

import evaluate_supervisor as ev
import train_hybrid_supervisor_v2 as v2
import finalize_conditional_normal_dynamics_expert as base
import finalize_conditional_normal_dynamics_expert_v2 as expert
import evaluate_conditional_normal_expert_seed17 as validation
import evaluate_conditional_normal_expert_scales_seed17 as sweep


OUT = "/root/gpufree-data/supervisor-results/conditional_normal_dynamics_expert_temporal_seed17.json"


def rolling(records, values, window):
    return ev.rolling_mean(np.asarray(values, dtype=float), records, window)


def thresholds_from_oof(records, rolling_score, percentile):
    by_task = {task: [] for task in range(10)}
    for key, indices in ev.episode_groups(records).items():
        by_task[key[0]].append(float(np.nanmax(rolling_score[indices])))
    return {task: float(np.percentile(by_task[task], percentile)) for task in range(10)}


def evaluate(records, rolling_score, thresholds, active):
    alarm = np.asarray([
        np.isfinite(value) and value > thresholds[record["task_id"]]
        for value, record in zip(rolling_score, records)
    ])
    groups = ev.episode_groups(records)
    result = {"episodes": len(groups), "alarm_episodes": sum(any(alarm[i] for i in ix) for ix in groups.values()),
              "step_alarm_rate": float(alarm.mean())}
    if active.any():
        for record, flag in zip(records, active): record["disturbance_active"] = bool(flag)
        result.update({
            "detected_episodes": sum(any(alarm[i] and active[i] for i in ix) for ix in groups.values()),
            "active_recall": float(alarm[active].mean()), "outside_rate": float(alarm[~active].mean()),
            "detection": ev.detection_delays(records, alarm.astype(float), 0.5),
        })
    return result


def main():
    torch.set_num_threads(4)
    checkpoint = torch.load(validation.MODEL, map_location="cpu")
    members = validation.load_members(checkpoint)
    # Reconstruct OOF normal scores: fold member never trained on the episode it predicts.
    all_records, all_x, all_y, all_folds = [], [], [], []
    for path, offset in ((v2.BASE, 0), (v2.NORMAL, 5), (v2.FINAL_NORMAL, 15)):
        _, records, x, y = base.load(path); all_records.extend(records); all_x.append(x); all_y.append(y)
        all_folds.extend([(offset + r["episode_idx"]) % 5 for r in records])
    all_x, all_y, all_folds = np.r_[tuple(all_x)], np.r_[tuple(all_y)], np.asarray(all_folds)
    mean = np.zeros_like(all_y); variance = np.zeros_like(all_y)
    for fold, member in enumerate(members):
        keep = all_folds == fold; mean[keep], variance[keep] = expert.member_predict(member, all_x[keep])
    oof_score, _ = expert.score(all_records, all_y, mean, variance, checkpoint["variance_scale"])

    datasets = {"normal": sweep.NORMAL, **sweep.ANOMALIES}
    predictions = {}
    for name, path in datasets.items():
        rows, records, x, y = base.load(path); m, var = expert.ensemble_predict(members, x)
        values, _ = expert.score(records, y, m, var, checkpoint["variance_scale"])
        predictions[name] = (rows, records, values)

    report = {"normal_only_calibration": True, "profiles": {}}
    for window in (3, 5):
        oof_rolling = rolling(all_records, oof_score, window)
        for percentile in (90, 95):
            name = f"w{window}_q{percentile}"
            thresholds = thresholds_from_oof(all_records, oof_rolling, percentile)
            profile = {"window": window, "percentile": percentile, "thresholds": thresholds, "results": {}}
            for dataset, (rows, records, values) in predictions.items():
                active = v2.active_flags(rows, records) if dataset != "normal" else np.zeros(len(records), bool)
                profile["results"][dataset] = evaluate(records, rolling(records, values, window), thresholds, active)
            report["profiles"][name] = profile
    pathlib.Path(OUT).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({name: value["results"] for name, value in report["profiles"].items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
