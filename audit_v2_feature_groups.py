#!/usr/bin/env python3
"""Feature-group and decision-stage audit for frozen-data v2; pilot labels are evaluation-only."""
import json
import pathlib

import numpy as np

import evaluate_supervisor as ev
import normal_mechanism_analysis as na
import train_hybrid_supervisor_v2 as v2


PILOT_NORMAL = "/root/gpufree-data/libero-traces/pilot_paired_normal_seed14_noise20260901_2ep.jsonl"
PILOT_ANOMALY = "/root/gpufree-data/libero-traces/pilot_random_multionset_scale050_seed14_2ep.jsonl"
OUT = "/root/gpufree-data/supervisor-results/v2_feature_group_audit_random_pilot.json"

# physical_features layout, verified from source; total dimension = 43.
GROUPS = {
    "prediction": np.arange(0, 3),
    "actual": np.arange(3, 6),
    "residual": np.arange(6, 9),
    "standardized_shortfall": np.arange(9, 10),
    "instant_handcrafted": np.arange(10, 21),
    "rolling_history": np.arange(21, 33),
    "task_onehot": np.arange(33, 43),
}


def trigger_values(records, probabilities):
    result = {}
    for key, indices in ev.episode_groups(records).items():
        values = []
        for j in range(len(indices)):
            window = indices[max(0, j - 2) : j + 1]
            if len(window) >= 2:
                values.append(float(np.partition(probabilities[window], -2)[-2]))
        result[key] = max(values) if values else 0.0
    return result


def build_training():
    _, (br, _, bx, by, _) = v2.load(v2.BASE)
    _, (nr, _, nx, ny, _) = v2.load(v2.NORMAL)
    _, (fr, _, fx, fy, _) = v2.load(v2.FINAL_NORMAL)
    nep = np.asarray([r["episode_idx"] for r in nr])
    dyntrain = nep <= 5
    cal = (nep >= 6) & (nep <= 7)
    model = na.fit_ridge(np.r_[bx, nx[dyntrain], fx], np.r_[by, ny[dyntrain], fy], nx[cal], ny[cal])
    cp = na.predict(model, nx[cal])
    cov, _ = ev.covariance_inverse(ny[cal] - cp)
    bfeat = v2.physical_features(model, cov, br, bx, by)
    nfeat = v2.physical_features(model, cov, nr, nx, ny)
    ffeat = v2.physical_features(model, cov, fr, fx, fy)
    trainx = [bfeat, nfeat[nep <= 5], ffeat]
    trainy = [np.zeros(len(br), int), np.zeros(int((nep <= 5).sum()), int), np.zeros(len(fr), int)]
    for path, final_path in zip(v2.ANOM, v2.FINAL_ANOM):
        rows, (rec, _, x, y, _) = v2.load(path)
        active = v2.active_flags(rows, rec)
        ep = np.asarray([r["episode_idx"] for r in rec])
        feat = v2.physical_features(model, cov, rec, x, y)
        trainx.append(feat[ep <= 1]); trainy.append(active[ep <= 1].astype(int))
        rows, (rec, _, x, y, _) = v2.load(final_path)
        active = v2.active_flags(rows, rec)
        ep = np.asarray([r["episode_idx"] for r in rec])
        feat = v2.physical_features(model, cov, rec, x, y)
        trainx.append(feat[ep == 0]); trainy.append(active[ep == 0].astype(int))
    normal_refs = [(br, bfeat), (nr, nfeat), (fr, ffeat)]
    return model, cov, np.r_[tuple(trainx)], np.r_[tuple(trainy)], normal_refs


def load_eval(path, model, cov, anomaly):
    rows, (rec, _, x, y, _) = v2.load(path)
    feat = v2.physical_features(model, cov, rec, x, y)
    active = v2.active_flags(rows, rec) if anomaly else np.zeros(len(rec), bool)
    return rec, feat, active


def evaluate(rec, features, active, predict, thresholds):
    probability = predict(features)
    normalized = np.asarray([p / (thresholds[r["task_id"]] + 1e-12) for p, r in zip(probability, rec)])
    point = normalized > 1.0
    alarm = ev.persistence_series(normalized, rec, 1.0, required=2, window=3)
    groups = ev.episode_groups(rec)
    alarm_episodes = sum(any(alarm[i] for i in indices) for indices in groups.values())
    result = {
        "episodes": len(groups),
        "alarm_episodes": alarm_episodes,
        "step_alarm_rate": float(alarm.mean()),
    }
    if active.any():
        result.update({
            "point_active_recall_before_persistence": float(point[active].mean()),
            "persistent_active_recall": float(alarm[active].mean()),
            "point_detected_episodes": sum(any(point[i] and active[i] for i in ix) for ix in groups.values()),
            "persistent_detected_episodes": sum(any(alarm[i] and active[i] for i in ix) for ix in groups.values()),
            "outside_persistent_rate": float(alarm[~active].mean()),
        })
    return result


def main():
    model, cov, trainx, trainy, normal_refs = build_training()
    nrec, nfeat, nactive = load_eval(PILOT_NORMAL, model, cov, False)
    arec, afeat, aactive = load_eval(PILOT_ANOMALY, model, cov, True)
    all_indices = np.arange(trainx.shape[1])
    variants = {"all": all_indices}
    for name, indices in GROUPS.items():
        variants[f"without_{name}"] = np.setdiff1d(all_indices, indices)
    variants["only_prediction_residual"] = np.arange(0, 10)
    variants["only_handcrafted_history_task"] = np.arange(10, 43)

    report = {"feature_dim": int(trainx.shape[1]), "groups": {k: v.tolist() for k, v in GROUPS.items()}, "variants": {}}
    for name, keep in variants.items():
        predict, loss = v2.train_classifier(trainx[:, keep], trainy)
        trigger_by_task = {task: [] for task in range(10)}
        for records, features in normal_refs:
            for key, value in trigger_values(records, predict(features[:, keep])).items():
                trigger_by_task[key[0]].append(value)
        thresholds = {task: float(np.percentile(trigger_by_task[task], 95)) for task in range(10)}
        report["variants"][name] = {
            "kept_dim": int(len(keep)),
            "train_loss": loss,
            "normal": evaluate(nrec, nfeat[:, keep], nactive, predict, thresholds),
            "anomaly": evaluate(arec, afeat[:, keep], aactive, predict, thresholds),
        }
        print(name, json.dumps(report["variants"][name], ensure_ascii=False))
    pathlib.Path(OUT).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
