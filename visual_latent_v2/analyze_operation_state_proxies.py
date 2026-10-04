"""Compare interpretable robot-state proxies across remaining error groups."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def proxy_matrix(data):
    mean = data["condition_mean"].astype(np.float64)
    last = data["condition_last"].astype(np.float64)
    # See dynamics_feature_schema.py. These are observable robot-side proxies,
    # not claims about semantic contact or object grasp success.
    return {
        "target_translation_norm_mean_m": mean[:, 3],
        "target_translation_norm_last_m": last[:, 3],
        "gripper_command_mean": mean[:, 7],
        "gripper_command_last": last[:, 7],
        "gripper_aperture_proxy_mean": np.linalg.norm(mean[:, 29:31], axis=1),
        "gripper_speed_mean": np.linalg.norm(mean[:, 31:33], axis=1),
        "joint_speed_mean": np.linalg.norm(mean[:, 15:22], axis=1),
        "previous_eef_speed_mean_mps": mean[:, 52],
        "command_turn_cosine_mean": mean[:, 50],
        "velocity_alignment_cosine_mean": mean[:, 51],
        "normal_minus_abnormal_logp": data["normal_logp_mean"] - data["abnormal_logp_mean"],
    }


def summary(values):
    if len(values) == 0:
        return None
    return {"n": int(len(values)), "q10": float(np.quantile(values, .1)),
            "median": float(np.median(values)), "q90": float(np.quantile(values, .9))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--errors", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    data = np.load(args.data, allow_pickle=False)
    errors = json.loads(args.errors.read_text(encoding="utf-8"))
    false_keys = {(r["episode_id"], int(r["chunk_id"]))
                  for r in errors["visual_false_trigger_chunks"]}
    missed_episodes = {r["episode_id"] for r in errors["base_missed_abnormal_episodes"]}
    eligible = data["previous_ambiguous_any"].astype(bool)
    active = data["previous_active_any"].astype(bool)
    clean = np.isclose(data["scale"], 1.0)
    false_visual = np.asarray([(str(e), int(c)) in false_keys
                               for e, c in zip(data["episode_id"], data["chunk_id"])])
    missed_active = np.asarray([str(e) in missed_episodes for e in data["episode_id"]]) & active & eligible
    detected_active = active & eligible & ~missed_active
    groups = {
        "clean_eligible_all": clean & eligible,
        "visual_false_trigger": false_visual,
        "base_missed_active": missed_active,
        "base_detected_active": detected_active,
    }
    proxies = proxy_matrix(data)
    report = {
        "warning": "Robot-state proxies do not identify contact, object identity, or grasp success.",
        "group_counts": {name: int(mask.sum()) for name, mask in groups.items()},
        "features": {feature: {name: summary(values[mask]) for name, mask in groups.items()}
                     for feature, values in proxies.items()},
        "by_task_counts": {
            name: {str(task): int(np.sum(mask & (data["task_id"] == task))) for task in range(10)}
            for name, mask in groups.items()
        },
    }
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
