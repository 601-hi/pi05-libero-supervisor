#!/usr/bin/env python3
"""Normal-only heteroscedastic dynamics expert and random-onset anomaly evaluation."""
import copy
import json
import pathlib

import numpy as np
import torch

import evaluate_supervisor as ev
import normal_mechanism_analysis as na
import train_hybrid_supervisor_v2 as v2


PILOT_NORMAL = "/root/gpufree-data/libero-traces/pilot_paired_normal_seed14_noise20260901_2ep.jsonl"
PILOT_ANOMALY = "/root/gpufree-data/libero-traces/pilot_random_multionset_scale050_seed14_2ep.jsonl"
OUT = "/root/gpufree-data/supervisor-results/conditional_normal_dynamics_expert_pilot.json"


def add_task(x, records):
    task = np.zeros((len(records), 10), dtype=float)
    task[np.arange(len(records)), [r["task_id"] for r in records]] = 1.0
    return np.c_[x, task]


def select(records, expression):
    return np.asarray([expression(r) for r in records], dtype=bool)


def fit_expert(x_train, y_train, x_validation, y_validation):
    torch.manual_seed(20260901)
    torch.set_num_threads(4)
    x_mean, x_scale = x_train.mean(0), x_train.std(0)
    x_scale[x_scale < 1e-6] = 1.0
    y_mean, y_scale = y_train.mean(0), y_train.std(0)
    y_scale[y_scale < 1e-6] = 1.0
    tx = torch.tensor((x_train - x_mean) / x_scale, dtype=torch.float32)
    ty = torch.tensor((y_train - y_mean) / y_scale, dtype=torch.float32)
    vx = torch.tensor((x_validation - x_mean) / x_scale, dtype=torch.float32)
    vy = torch.tensor((y_validation - y_mean) / y_scale, dtype=torch.float32)
    net = torch.nn.Sequential(
        torch.nn.Linear(x_train.shape[1], 128), torch.nn.SiLU(),
        torch.nn.Linear(128, 128), torch.nn.SiLU(),
        torch.nn.Linear(128, 6),
    )
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=2e-4)

    def nll(output, target):
        mean, logvar = output[:, :3], output[:, 3:].clamp(-6.0, 3.0)
        return 0.5 * (logvar + (target - mean).square() * torch.exp(-logvar)).sum(1).mean()

    best = None
    for epoch in range(600):
        net.train(); optimizer.zero_grad(); loss = nll(net(tx), ty); loss.backward(); optimizer.step()
        if epoch % 10 == 0:
            net.eval()
            with torch.no_grad(): validation = float(nll(net(vx), vy))
            if best is None or validation < best[0]:
                best = (validation, epoch, copy.deepcopy(net.state_dict()))
    net.load_state_dict(best[2]); net.eval()

    def predict_score(x, y):
        with torch.no_grad():
            output = net(torch.tensor((x - x_mean) / x_scale, dtype=torch.float32)).numpy()
        mean_z = output[:, :3]
        logvar = np.clip(output[:, 3:], -6.0, 3.0)
        target_z = (y - y_mean) / y_scale
        innovation = (target_z - mean_z) / np.exp(0.5 * logvar)
        score = np.sum(innovation ** 2, axis=1)
        prediction = mean_z * y_scale + y_mean
        return prediction, innovation, score

    return predict_score, {"best_validation_nll": best[0], "best_epoch": best[1]}


def load(path):
    rows, (records, _, x, y, _) = v2.load(path)
    return rows, records, add_task(x, records), y


def episode_trigger_values(records, score):
    result = {}
    for key, indices in ev.episode_groups(records).items():
        values = []
        for j in range(len(indices)):
            window = indices[max(0, j - 2):j + 1]
            if len(window) >= 2:
                values.append(float(np.partition(score[window], -2)[-2]))
        result[key] = max(values) if values else 0.0
    return result


def evaluate(records, score, thresholds, active):
    normalized = np.asarray([s / (thresholds[r["task_id"]] + 1e-12) for s, r in zip(score, records)])
    point = normalized > 1.0
    alarm = ev.persistence_series(normalized, records, 1.0, required=2, window=3)
    groups = ev.episode_groups(records)
    result = {
        "episodes": len(groups), "steps": len(records),
        "alarm_episodes": sum(any(alarm[i] for i in ix) for ix in groups.values()),
        "step_alarm_rate": float(alarm.mean()),
    }
    if active.any():
        for r, a in zip(records, active): r["disturbance_active"] = bool(a)
        result.update({
            "point_detected_episodes": sum(any(point[i] and active[i] for i in ix) for ix in groups.values()),
            "persistent_detected_episodes": sum(any(alarm[i] and active[i] for i in ix) for ix in groups.values()),
            "point_active_recall": float(point[active].mean()),
            "persistent_active_recall": float(alarm[active].mean()),
            "outside_alarm_rate": float(alarm[~active].mean()),
            "detection": ev.detection_delays(records, alarm.astype(float), 0.5),
        })
    return result


def main():
    datasets = {}
    for name, path in (("base", v2.BASE), ("independent", v2.NORMAL), ("final", v2.FINAL_NORMAL)):
        datasets[name] = load(path)
    train_parts, train_y, val_parts, val_y, cal_refs = [], [], [], [], []
    # Episode-level splits preserve three independent seeds in both train and calibration.
    split_limits = {"base": 3, "independent": 5, "final": 3}
    for name, (_, records, x, y) in datasets.items():
        cutoff = split_limits[name]
        train_mask = select(records, lambda r, c=cutoff: r["episode_idx"] <= c)
        cal_mask = ~train_mask
        train_parts.append(x[train_mask]); train_y.append(y[train_mask])
        val_parts.append(x[cal_mask]); val_y.append(y[cal_mask])
        cal_refs.append(([r for r, k in zip(records, cal_mask) if k], x[cal_mask], y[cal_mask]))
    predict_score, fit = fit_expert(np.r_[tuple(train_parts)], np.r_[tuple(train_y)], np.r_[tuple(val_parts)], np.r_[tuple(val_y)])

    trigger_by_task = {task: [] for task in range(10)}
    calibration_rmse = []
    for records, x, y in cal_refs:
        prediction, _, score = predict_score(x, y)
        calibration_rmse.append(float(np.sqrt(np.mean((prediction - y) ** 2))))
        for key, value in episode_trigger_values(records, score).items():
            trigger_by_task[key[0]].append(value)
    thresholds = {task: float(np.percentile(trigger_by_task[task], 95)) for task in range(10)}

    _, nrec, nx, ny = load(PILOT_NORMAL)
    _, _, nscore = predict_score(nx, ny)
    arows, arec, ax, ay = load(PILOT_ANOMALY)
    _, _, ascore = predict_score(ax, ay)
    active = v2.active_flags(arows, arec)
    report = {
        "model": "normal-only heteroscedastic MLP dynamics expert",
        "inputs": "deployable pre-step dynamic features plus task one-hot",
        "fit": fit,
        "calibration_rmse_m": calibration_rmse,
        "thresholds": thresholds,
        "normal": evaluate(nrec, nscore, thresholds, np.zeros(len(nrec), bool)),
        "random_onset_scale050": evaluate(arec, ascore, thresholds, active),
    }
    pathlib.Path(OUT).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
