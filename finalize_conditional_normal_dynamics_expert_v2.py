#!/usr/bin/env python3
"""Five-fold out-of-fold calibrated, normal-only dynamics expert ensemble."""
import json
import pathlib

import numpy as np
import torch

import evaluate_supervisor as ev
import train_hybrid_supervisor_v2 as v2
import finalize_conditional_normal_dynamics_expert as base


OUT = "/root/gpufree-data/supervisor-results/conditional_normal_dynamics_expert_frozen_v2.json"
MODEL_OUT = "/root/gpufree-data/supervisor-results/conditional_normal_dynamics_expert_frozen_v2.pt"
FOLD_SEEDS = (20260911, 20260912, 20260913, 20260914, 20260915)
EPOCHS = 800
EPS = 1e-12


def train_member(seed, x, y):
    torch.manual_seed(seed)
    x_mean, x_scale = x.mean(0), x.std(0); x_scale[x_scale < 1e-6] = 1.0
    y_mean, y_scale = y.mean(0), y.std(0); y_scale[y_scale < 1e-6] = 1.0
    tx = torch.tensor((x - x_mean) / x_scale, dtype=torch.float32)
    ty = torch.tensor((y - y_mean) / y_scale, dtype=torch.float32)
    net = base.Expert(x.shape[1])
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=2e-4)
    for _ in range(EPOCHS):
        net.train(); optimizer.zero_grad(); loss = base.gaussian_nll(net(tx), ty); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0); optimizer.step()
    net.eval()
    return {"net": net, "x_mean": x_mean, "x_scale": x_scale, "y_mean": y_mean, "y_scale": y_scale,
            "seed": seed, "train_nll": float(loss)}


def member_predict(member, x):
    tensor = torch.tensor((x - member["x_mean"]) / member["x_scale"], dtype=torch.float32)
    with torch.no_grad(): output = member["net"](tensor).numpy()
    mean = output[:, :3] * member["y_scale"] + member["y_mean"]
    variance = np.exp(np.clip(output[:, 3:], -6.0, 3.0)) * (member["y_scale"] ** 2)
    return mean, variance


def ensemble_predict(members, x):
    predictions = [member_predict(member, x) for member in members]
    means = np.asarray([item[0] for item in predictions])
    variances = np.asarray([item[1] for item in predictions])
    return means.mean(0), variances.mean(0) + means.var(0)


def score(records, y, mean, variance, variance_scale):
    scale = np.asarray([variance_scale[r["task_id"]] for r in records])
    calibrated = np.maximum(variance * scale, 1e-12)
    innovation = (y - mean) / np.sqrt(calibrated)
    return np.sum(innovation ** 2, axis=1), innovation


def main():
    torch.set_num_threads(4)
    records_all, x_all, y_all, folds_all = [], [], [], []
    source_offsets = ((v2.BASE, 0), (v2.NORMAL, 5), (v2.FINAL_NORMAL, 15))
    for path, offset in source_offsets:
        _, records, x, y = base.load(path)
        records_all.extend(records); x_all.append(x); y_all.append(y)
        folds_all.extend([(offset + r["episode_idx"]) % 5 for r in records])
    x_all, y_all, folds_all = np.r_[tuple(x_all)], np.r_[tuple(y_all)], np.asarray(folds_all)

    members = []
    oof_mean = np.zeros_like(y_all); oof_variance = np.zeros_like(y_all)
    for fold, seed in enumerate(FOLD_SEEDS):
        train = folds_all != fold; heldout = ~train
        member = train_member(seed, x_all[train], y_all[train]); members.append(member)
        oof_mean[heldout], oof_variance[heldout] = member_predict(member, x_all[heldout])

    ratios = {task: [] for task in range(10)}
    for index, record in enumerate(records_all):
        ratios[record["task_id"]].append((y_all[index] - oof_mean[index]) ** 2 / np.maximum(oof_variance[index], 1e-12))
    variance_scale = {
        task: np.clip(np.mean(np.asarray(ratios[task]), axis=0), 0.25, 25.0) for task in range(10)
    }
    oof_score, oof_innovation = score(records_all, y_all, oof_mean, oof_variance, variance_scale)
    triggers = base.trigger_values(records_all, oof_score)
    trigger_by_task = {task: [value for (t, _), value in triggers.items() if t == task] for task in range(10)}
    threshold_sets = {
        str(percentile): {task: float(np.percentile(trigger_by_task[task], percentile)) for task in range(10)}
        for percentile in (80, 85, 90, 92.5, 95, 97.5)
    }
    thresholds = threshold_sets["95"]

    nrows, nrec, nx, ny = base.load(base.PILOT_NORMAL)
    nmean, nvariance = ensemble_predict(members, nx); nscore, ninnovation = score(nrec, ny, nmean, nvariance, variance_scale)
    arows, arec, ax, ay = base.load(base.PILOT_ANOMALY)
    amean, avariance = ensemble_predict(members, ax); ascore, _ = score(arec, ay, amean, avariance, variance_scale)
    active = v2.active_flags(arows, arec)
    operating_points = {}
    for name, operating_thresholds in threshold_sets.items():
        operating_points[name] = {
            "normal": base.evaluate(nrec, nscore, operating_thresholds, np.zeros(len(nrec), bool)),
            "random_scale050": base.evaluate(arec, ascore, operating_thresholds, active),
        }
    report = {
        "status": "frozen_v2_oof",
        "training": {"normal_only": True, "normal_episodes_per_task": 20, "folds": 5, "epochs": EPOCHS,
                     "fold_seeds": list(FOLD_SEEDS), "steps": len(x_all),
                     "member_train_nll": [m["train_nll"] for m in members]},
        "oof_calibration": {"episodes_per_task": 20, "episode_trigger_percentile": 95,
                            "persistence": "2of3", "thresholds": thresholds,
                            "threshold_sets": threshold_sets,
                            "axis_95_coverage": float((np.abs(oof_innovation) <= 1.96).mean()),
                            "variance_scale_by_task_axis": {str(k): v.tolist() for k, v in variance_scale.items()}},
        "pilot_normal": base.evaluate(nrec, nscore, thresholds, np.zeros(len(nrec), bool)),
        "pilot_random_scale050": base.evaluate(arec, ascore, thresholds, active),
        "pilot_normal_axis_95_coverage": float((np.abs(ninnovation) <= 1.96).mean()),
        "operating_points_evaluation_only": operating_points,
    }
    serial_members = [{
        "state_dict": m["net"].state_dict(), "x_mean": m["x_mean"], "x_scale": m["x_scale"],
        "y_mean": m["y_mean"], "y_scale": m["y_scale"], "seed": m["seed"]
    } for m in members]
    torch.save({"members": serial_members, "input_dim": x_all.shape[1], "variance_scale": variance_scale,
                "thresholds": thresholds, "threshold_sets": threshold_sets,
                "persistence": {"required": 2, "window": 3},
                "observability": {"episode_mean_target_norm_warning_below_m": 0.015}}, MODEL_OUT)
    pathlib.Path(OUT).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "training": report["training"], "oof_calibration": report["oof_calibration"],
        "pilot_normal": {k: v for k, v in report["pilot_normal"].items() if k != "episodes_detail"},
        "pilot_random_scale050": {k: v for k, v in report["pilot_random_scale050"].items() if k != "episodes_detail"},
        "pilot_normal_axis_95_coverage": report["pilot_normal_axis_95_coverage"],
        "operating_points": {
            name: {
                "normal": {k: v for k, v in value["normal"].items() if k != "episodes_detail"},
                "random": {k: v for k, v in value["random_scale050"].items() if k != "episodes_detail"},
            }
            for name, value in operating_points.items()
        },
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
