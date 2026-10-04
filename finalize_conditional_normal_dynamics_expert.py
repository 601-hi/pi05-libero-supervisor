#!/usr/bin/env python3
"""Freeze a calibrated ensemble normal-dynamics expert using normal data only."""
import copy
import json
import pathlib

import numpy as np
import torch

import evaluate_supervisor as ev
import train_hybrid_supervisor_v2 as v2


PILOT_NORMAL = "/root/gpufree-data/libero-traces/pilot_paired_normal_seed14_noise20260901_2ep.jsonl"
PILOT_ANOMALY = "/root/gpufree-data/libero-traces/pilot_random_multionset_scale050_seed14_2ep.jsonl"
OUT = "/root/gpufree-data/supervisor-results/conditional_normal_dynamics_expert_frozen_v1.json"
MODEL_OUT = "/root/gpufree-data/supervisor-results/conditional_normal_dynamics_expert_frozen_v1.pt"
ENSEMBLE_SEEDS = (20260901, 20260902, 20260903, 20260904, 20260905)
EPS = 1e-12


def load(path):
    rows, (records, _, x, y, _) = v2.load(path)
    task = np.zeros((len(records), 10), dtype=float)
    task[np.arange(len(records)), [r["task_id"] for r in records]] = 1.0
    return rows, records, np.c_[x, task], y


def mask(records, predicate):
    return np.asarray([predicate(r) for r in records], dtype=bool)


class Expert(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 128), torch.nn.SiLU(),
            torch.nn.Linear(128, 128), torch.nn.SiLU(),
            torch.nn.Linear(128, 6),
        )

    def forward(self, x):
        return self.net(x)


def gaussian_nll(output, target):
    mean, logvar = output[:, :3], output[:, 3:].clamp(-6.0, 3.0)
    return 0.5 * (logvar + (target - mean).square() * torch.exp(-logvar)).sum(1).mean()


def train_member(seed, tx, ty, vx, vy):
    torch.manual_seed(seed)
    net = Expert(tx.shape[1])
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=2e-4)
    best = None
    for epoch in range(801):
        net.train(); optimizer.zero_grad(); loss = gaussian_nll(net(tx), ty); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        optimizer.step()
        if epoch % 10 == 0:
            net.eval()
            with torch.no_grad(): val = float(gaussian_nll(net(vx), vy))
            if best is None or val < best[0]: best = (val, epoch, copy.deepcopy(net.state_dict()))
    net.load_state_dict(best[2]); net.eval()
    return net, {"seed": seed, "best_validation_nll": best[0], "best_epoch": best[1]}


def ensemble_predict(nets, x, x_mean, x_scale, y_mean, y_scale):
    tensor = torch.tensor((x - x_mean) / x_scale, dtype=torch.float32)
    means, variances = [], []
    with torch.no_grad():
        for net in nets:
            output = net(tensor).numpy()
            means.append(output[:, :3])
            variances.append(np.exp(np.clip(output[:, 3:], -6.0, 3.0)))
    means = np.asarray(means)
    variances = np.asarray(variances)
    mean_z = means.mean(0)
    # Total predictive variance = mean aleatoric variance + ensemble epistemic variance.
    variance_z = variances.mean(0) + means.var(0)
    return mean_z * y_scale + y_mean, variance_z * (y_scale ** 2)


def calibrated_score(records, y, mean, variance, variance_scale):
    scales = np.asarray([variance_scale[r["task_id"]] for r in records])
    calibrated_variance = np.maximum(variance * scales, 1e-12)
    innovation = (y - mean) / np.sqrt(calibrated_variance)
    return np.sum(innovation ** 2, axis=1), innovation, calibrated_variance


def trigger_values(records, score):
    result = {}
    for key, indices in ev.episode_groups(records).items():
        candidates = []
        for offset in range(len(indices)):
            window = indices[max(0, offset - 2):offset + 1]
            if len(window) >= 2: candidates.append(float(np.partition(score[window], -2)[-2]))
        result[key] = max(candidates) if candidates else 0.0
    return result


def evaluate(records, score, thresholds, active):
    normalized = np.asarray([s / (thresholds[r["task_id"]] + EPS) for s, r in zip(score, records)])
    point = normalized > 1.0
    alarm = ev.persistence_series(normalized, records, 1.0, required=2, window=3)
    groups = ev.episode_groups(records)
    details = []
    for key, indices in groups.items():
        active_indices = [i for i in indices if active[i]]
        point_hits = [records[i]["action_index"] for i in indices if point[i] and active[i]]
        alarm_hits = [records[i]["action_index"] for i in indices if alarm[i] and active[i]]
        details.append({
            "task_id": key[0], "episode_idx": key[1],
            "point_detected": bool(point_hits), "persistent_detected": bool(alarm_hits),
            "point_active_actions": point_hits, "persistent_active_actions": alarm_hits,
            "max_active_score_ratio": float(max((normalized[i] for i in active_indices), default=0.0)),
        })
    result = {
        "episodes": len(groups), "steps": len(records),
        "alarm_episodes": sum(any(alarm[i] for i in ix) for ix in groups.values()),
        "step_alarm_rate": float(alarm.mean()), "episodes_detail": details,
    }
    if active.any():
        for r, a in zip(records, active): r["disturbance_active"] = bool(a)
        result.update({
            "point_detected_episodes": sum(d["point_detected"] for d in details),
            "persistent_detected_episodes": sum(d["persistent_detected"] for d in details),
            "point_active_recall": float(point[active].mean()),
            "persistent_active_recall": float(alarm[active].mean()),
            "outside_alarm_rate": float(alarm[~active].mean()),
            "detection": ev.detection_delays(records, alarm.astype(float), 0.5),
        })
    return result


def main():
    torch.set_num_threads(4)
    sources = {name: load(path) for name, path in (
        ("seed7", v2.BASE), ("seed11", v2.NORMAL), ("seed13", v2.FINAL_NORMAL)
    )}
    train_x, train_y, validation_x, validation_y, calibration = [], [], [], [], []
    # Per task: train 13 episodes, validate 3, calibrate 4; no seed14 data enters fitting/calibration.
    for name, (_, records, x, y) in sources.items():
        if name == "seed7":
            tr = mask(records, lambda r: r["episode_idx"] <= 2)
            va = mask(records, lambda r: r["episode_idx"] == 3)
            ca = mask(records, lambda r: r["episode_idx"] == 4)
        elif name == "seed11":
            tr = mask(records, lambda r: r["episode_idx"] <= 5)
            va = mask(records, lambda r: 6 <= r["episode_idx"] <= 7)
            ca = mask(records, lambda r: r["episode_idx"] >= 8)
        else:
            tr = mask(records, lambda r: r["episode_idx"] <= 3)
            va = mask(records, lambda r: r["episode_idx"] == 4)
            ca = np.zeros(len(records), dtype=bool)
        train_x.append(x[tr]); train_y.append(y[tr])
        validation_x.append(x[va]); validation_y.append(y[va])
        if ca.any(): calibration.append(([r for r, k in zip(records, ca) if k], x[ca], y[ca]))
    train_x, train_y = np.r_[tuple(train_x)], np.r_[tuple(train_y)]
    validation_x, validation_y = np.r_[tuple(validation_x)], np.r_[tuple(validation_y)]
    x_mean, x_scale = train_x.mean(0), train_x.std(0); x_scale[x_scale < 1e-6] = 1.0
    y_mean, y_scale = train_y.mean(0), train_y.std(0); y_scale[y_scale < 1e-6] = 1.0
    tx = torch.tensor((train_x - x_mean) / x_scale, dtype=torch.float32)
    ty = torch.tensor((train_y - y_mean) / y_scale, dtype=torch.float32)
    vx = torch.tensor((validation_x - x_mean) / x_scale, dtype=torch.float32)
    vy = torch.tensor((validation_y - y_mean) / y_scale, dtype=torch.float32)
    nets, members = [], []
    for seed in ENSEMBLE_SEEDS:
        net, metadata = train_member(seed, tx, ty, vx, vy); nets.append(net); members.append(metadata)

    # Variance recalibration uses calibration residual / predicted variance separately for each task and axis.
    calibration_cache = []
    ratios = {task: [] for task in range(10)}
    for records, x, y in calibration:
        mean, variance = ensemble_predict(nets, x, x_mean, x_scale, y_mean, y_scale)
        for task in range(10):
            keep = np.asarray([r["task_id"] == task for r in records])
            if keep.any(): ratios[task].append(((y[keep] - mean[keep]) ** 2) / np.maximum(variance[keep], 1e-12))
        calibration_cache.append((records, y, mean, variance))
    variance_scale = {
        task: np.clip(np.mean(np.concatenate(ratios[task], axis=0), axis=0), 0.25, 16.0)
        for task in range(10)
    }
    trigger_by_task = {task: [] for task in range(10)}
    calibration_coverage = []
    for records, y, mean, variance in calibration_cache:
        score, innovation, _ = calibrated_score(records, y, mean, variance, variance_scale)
        calibration_coverage.append(float((np.abs(innovation) <= 1.96).mean()))
        for key, value in trigger_values(records, score).items(): trigger_by_task[key[0]].append(value)
    thresholds = {task: float(np.percentile(trigger_by_task[task], 95)) for task in range(10)}

    nrows, nrec, nx, ny = load(PILOT_NORMAL)
    nmean, nvar = ensemble_predict(nets, nx, x_mean, x_scale, y_mean, y_scale)
    nscore, ninnovation, _ = calibrated_score(nrec, ny, nmean, nvar, variance_scale)
    arows, arec, ax, ay = load(PILOT_ANOMALY)
    amean, avar = ensemble_predict(nets, ax, x_mean, x_scale, y_mean, y_scale)
    ascore, _, _ = calibrated_score(arec, ay, amean, avar, variance_scale)
    active = v2.active_flags(arows, arec)
    report = {
        "status": "frozen_v1",
        "training": {"normal_only": True, "train_steps": len(train_x), "validation_steps": len(validation_x),
                     "ensemble_seeds": list(ENSEMBLE_SEEDS), "members": members},
        "variance_scale_by_task_axis": {str(k): v.tolist() for k, v in variance_scale.items()},
        "calibration": {"episodes_per_task": len(trigger_by_task[0]), "coverage_95_per_source": calibration_coverage,
                        "episode_trigger_percentile": 95, "thresholds": thresholds, "persistence": "2of3"},
        "pilot_normal": evaluate(nrec, nscore, thresholds, np.zeros(len(nrec), bool)),
        "pilot_random_scale050": evaluate(arec, ascore, thresholds, active),
        "pilot_normal_axis_95_coverage": float((np.abs(ninnovation) <= 1.96).mean()),
    }
    torch.save({
        "model_state_dicts": [net.state_dict() for net in nets], "input_dim": train_x.shape[1],
        "x_mean": x_mean, "x_scale": x_scale, "y_mean": y_mean, "y_scale": y_scale,
        "variance_scale_by_task_axis": variance_scale, "thresholds": thresholds,
        "ensemble_seeds": ENSEMBLE_SEEDS, "persistence": {"required": 2, "window": 3},
    }, MODEL_OUT)
    pathlib.Path(OUT).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "training": report["training"], "calibration": report["calibration"],
        "pilot_normal": {k: v for k, v in report["pilot_normal"].items() if k != "episodes_detail"},
        "pilot_random_scale050": {k: v for k, v in report["pilot_random_scale050"].items() if k != "episodes_detail"},
        "pilot_normal_axis_95_coverage": report["pilot_normal_axis_95_coverage"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
