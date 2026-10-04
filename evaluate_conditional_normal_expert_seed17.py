#!/usr/bin/env python3
"""Independent seed17 evaluation of frozen conditional normal dynamics expert profiles."""
import json
import pathlib

import numpy as np
import torch

import evaluate_supervisor as ev
import evaluate_v2_seed14_frozen as old_v2
import train_hybrid_supervisor_v2 as v2
import finalize_conditional_normal_dynamics_expert as base
import finalize_conditional_normal_dynamics_expert_v2 as expert


NORMAL = "/root/gpufree-data/libero-traces/normal_expert_validation_normal_seed17_noise20260917_2ep.jsonl"
ANOMALY = "/root/gpufree-data/libero-traces/normal_expert_validation_random_scale050_seed17_noise20260917_2ep.jsonl"
MODEL = "/root/gpufree-data/supervisor-results/conditional_normal_dynamics_expert_frozen_v2.pt"
OUT = "/root/gpufree-data/supervisor-results/conditional_normal_dynamics_expert_seed17_validation.json"


def load_members(checkpoint):
    members = []
    for saved in checkpoint["members"]:
        net = base.Expert(checkpoint["input_dim"]); net.load_state_dict(saved["state_dict"]); net.eval()
        members.append({**saved, "net": net})
    return members


def evaluate_profile(records, score, thresholds, active, active_counts):
    report = base.evaluate(records, score, thresholds, active)
    detail = {(d["task_id"], d["episode_idx"]): d for d in report["episodes_detail"]}
    groups = ev.episode_groups(records)
    exposure = []
    for key, indices in groups.items():
        active_indices = [i for i in indices if active[i]]
        mean_target = float(np.mean([records[i]["target_norm"] for i in active_indices])) if active_indices else 0.0
        exposure.append({
            "task_id": key[0], "episode_idx": key[1], "active_steps": active_counts[key],
            "mean_active_target_norm_m": mean_target,
            "observable": mean_target >= 0.015,
            "persistent_detected": detail[key]["persistent_detected"],
            "max_active_score_ratio": detail[key]["max_active_score_ratio"],
        })
    full = [e for e in exposure if e["active_steps"] == 10]
    observable = [e for e in exposure if e["observable"]]
    report["exposure_summary"] = {
        "complete_10_step_episodes": len(full),
        "detected_complete_10_step": sum(e["persistent_detected"] for e in full),
        "observable_episodes_mean_target_ge_0p015m": len(observable),
        "detected_observable": sum(e["persistent_detected"] for e in observable),
    }
    report["exposure_detail"] = exposure
    return report


def main():
    torch.set_num_threads(4)
    checkpoint = torch.load(MODEL, map_location="cpu")
    members = load_members(checkpoint)
    nrows, nrec, nx, ny = base.load(NORMAL)
    nmean, nvar = expert.ensemble_predict(members, nx)
    nscore, _ = expert.score(nrec, ny, nmean, nvar, checkpoint["variance_scale"])
    arows, arec, ax, ay = base.load(ANOMALY)
    amean, avar = expert.ensemble_predict(members, ax)
    ascore, _ = expert.score(arec, ay, amean, avar, checkpoint["variance_scale"])
    active = v2.active_flags(arows, arec)
    active_counts = {}
    for key, indices in ev.episode_groups(arec).items(): active_counts[key] = int(active[indices].sum())

    profiles = {}
    for name in ("85", "90", "95"):
        thresholds = checkpoint["threshold_sets"][name]
        profiles[name] = {
            "normal": base.evaluate(nrec, nscore, thresholds, np.zeros(len(nrec), bool)),
            "anomaly": evaluate_profile(arec, ascore, thresholds, active, active_counts),
        }
    old_model, old_cov, old_predict, old_thresholds, old_loss = old_v2.fit_frozen()
    report = {
        "evaluation_seed": 17, "sampling_noise_seed": 20260917,
        "frozen_before_seed17": True,
        "profiles": profiles,
        "old_v2": {
            "normal": old_v2.evaluate(NORMAL, old_model, old_cov, old_predict, old_thresholds, False),
            "anomaly": old_v2.evaluate(ANOMALY, old_model, old_cov, old_predict, old_thresholds, True),
        },
    }
    pathlib.Path(OUT).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    compact = {"profiles": {}, "old_v2": {}}
    for name, value in profiles.items():
        compact["profiles"][name] = {
            "normal": {k: v for k, v in value["normal"].items() if k != "episodes_detail"},
            "anomaly": {k: v for k, v in value["anomaly"].items() if k not in ("episodes_detail", "exposure_detail")},
        }
    compact["old_v2"] = {
        key: {k: v for k, v in value.items() if k != "episodes_detail"} for key, value in report["old_v2"].items()
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
