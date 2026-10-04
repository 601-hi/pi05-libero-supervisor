#!/usr/bin/env python3
"""Frozen seed17 severity sweep for the finalized normal dynamics expert."""
import json
import pathlib

import numpy as np
import torch

import evaluate_supervisor as ev
import evaluate_v2_seed14_frozen as old_v2
import train_hybrid_supervisor_v2 as v2
import finalize_conditional_normal_dynamics_expert as base
import finalize_conditional_normal_dynamics_expert_v2 as expert
import evaluate_conditional_normal_expert_seed17 as validation


NORMAL = validation.NORMAL
ANOMALIES = {
    "scale025": "/root/gpufree-data/libero-traces/normal_expert_validation_random_scale025_seed17_noise20260917_2ep.jsonl",
    "scale050": validation.ANOMALY,
    "scale075": "/root/gpufree-data/libero-traces/normal_expert_validation_random_scale075_seed17_noise20260917_2ep.jsonl",
}
OUT = "/root/gpufree-data/supervisor-results/conditional_normal_dynamics_expert_seed17_severity_sweep.json"


def main():
    torch.set_num_threads(4)
    checkpoint = torch.load(validation.MODEL, map_location="cpu")
    members = validation.load_members(checkpoint)
    _, nrec, nx, ny = base.load(NORMAL)
    nmean, nvar = expert.ensemble_predict(members, nx)
    nscore, _ = expert.score(nrec, ny, nmean, nvar, checkpoint["variance_scale"])
    old_model, old_cov, old_predict, old_thresholds, _ = old_v2.fit_frozen()
    report = {"evaluation_seed": 17, "frozen_before_severity_sweep": True, "profiles": {}, "old_v2": {}}
    for profile in ("85", "90", "95"):
        thresholds = checkpoint["threshold_sets"][profile]
        report["profiles"][profile] = {
            "normal": base.evaluate(nrec, nscore, thresholds, np.zeros(len(nrec), bool)), "anomalies": {}
        }
        for name, path in ANOMALIES.items():
            rows, records, x, y = base.load(path)
            mean, variance = expert.ensemble_predict(members, x)
            anomaly_score, _ = expert.score(records, y, mean, variance, checkpoint["variance_scale"])
            active = v2.active_flags(rows, records)
            counts = {key: int(active[indices].sum()) for key, indices in ev.episode_groups(records).items()}
            report["profiles"][profile]["anomalies"][name] = validation.evaluate_profile(
                records, anomaly_score, thresholds, active, counts
            )
    report["old_v2"]["normal"] = old_v2.evaluate(NORMAL, old_model, old_cov, old_predict, old_thresholds, False)
    for name, path in ANOMALIES.items():
        report["old_v2"][name] = old_v2.evaluate(path, old_model, old_cov, old_predict, old_thresholds, True)
    pathlib.Path(OUT).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    compact = {"profiles": {}, "old_v2": {}}
    for profile, value in report["profiles"].items():
        compact["profiles"][profile] = {
            "normal": {k: v for k, v in value["normal"].items() if k != "episodes_detail"},
            "anomalies": {
                name: {k: v for k, v in result.items() if k not in ("episodes_detail", "exposure_detail")}
                for name, result in value["anomalies"].items()
            },
        }
    compact["old_v2"] = {
        name: {k: v for k, v in value.items() if k != "episodes_detail"} for name, value in report["old_v2"].items()
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
