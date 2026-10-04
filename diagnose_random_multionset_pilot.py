#!/usr/bin/env python3
"""Paired causal diagnosis of the seed-14 random-onset pilot with the frozen v2 supervisor."""
import collections
import json
import pathlib

import numpy as np

import evaluate_v2_seed14_frozen as frozen


NORMAL = "/root/gpufree-data/libero-traces/pilot_paired_normal_seed14_noise20260901_2ep.jsonl"
DISTURBED = "/root/gpufree-data/libero-traces/pilot_random_multionset_scale050_seed14_2ep.jsonl"
OUT = "/root/gpufree-data/supervisor-results/v2_random_multionset_paired_diagnosis_seed14.json"


def load(path):
    rows = [json.loads(line) for line in pathlib.Path(path).open(encoding="utf-8") if line.strip()]
    groups = collections.defaultdict(list)
    ends = {}
    for row in rows:
        if row["event"] == "step":
            groups[(row["task_id"], row["episode_idx"])].append(row)
        elif row["event"] == "episode_end":
            ends[(row["task_id"], row["episode_idx"])] = row
    return rows, groups, ends


def vec(row, name):
    return np.asarray(row[name], dtype=float)


def progress(row):
    target = vec(row, "intended_target_translation")
    actual = vec(row, "actual_translation")
    return float(np.dot(target, actual) / (np.dot(target, target) + 1e-12))


def summarize(values):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return None
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def main():
    _, normal, normal_ends = load(NORMAL)
    _, disturbed, disturbed_ends = load(DISTURBED)
    model, cov, predict, thresholds, loss = frozen.fit_frozen()
    normal_eval = frozen.evaluate(NORMAL, model, cov, predict, thresholds, False)
    disturbed_eval = frozen.evaluate(DISTURBED, model, cov, predict, thresholds, True)
    detection = {
        (e["task_id"], e["episode_idx"]): e for e in disturbed_eval["episodes_detail"]
    }

    episodes = []
    for key in sorted(disturbed):
        base = normal[key]
        fault = disturbed[key]
        start = int(disturbed_ends[key]["actual_disturbance_start"])
        common = min(len(base), len(fault))
        pre = range(min(start, common))
        pre_action_max = max(
            (float(np.max(np.abs(vec(base[i], "intended_action") - vec(fault[i], "intended_action")))) for i in pre),
            default=0.0,
        )
        pre_eef_max = max(
            (float(np.linalg.norm(vec(base[i], "eef_pos_after") - vec(fault[i], "eef_pos_after"))) for i in pre),
            default=0.0,
        )
        first_action_difference = next(
            (
                i
                for i in range(common)
                if np.max(np.abs(vec(base[i], "intended_action") - vec(fault[i], "intended_action"))) > 1e-12
            ),
            None,
        )
        same_intended = 0
        for i in range(start, min(start + 10, common)):
            if np.max(np.abs(vec(base[i], "intended_action") - vec(fault[i], "intended_action"))) <= 1e-12:
                same_intended += 1
            else:
                break
        same_range = range(start, min(start + same_intended, common))
        active_range = range(start, min(start + 10, common))
        end_index = min(start + 9, common - 1)
        detail = detection[key]
        episodes.append(
            {
                "task_id": key[0],
                "episode_idx": key[1],
                "start": start,
                "detected": detail["detected"],
                "alarm_actions": detail["alarm_actions"],
                "pre_action_max_abs_difference": pre_action_max,
                "pre_eef_after_max_difference_m": pre_eef_max,
                "first_intended_action_difference": first_action_difference,
                "same_intended_steps_after_onset": same_intended,
                "same_intended_normal_progress": summarize([progress(base[i]) for i in same_range]),
                "same_intended_disturbed_progress": summarize([progress(fault[i]) for i in same_range]),
                "active_actual_translation_difference_m": summarize(
                    [np.linalg.norm(vec(base[i], "actual_translation") - vec(fault[i], "actual_translation")) for i in active_range]
                ),
                "eef_separation_at_active_end_m": float(
                    np.linalg.norm(vec(base[end_index], "eef_pos_after") - vec(fault[end_index], "eef_pos_after"))
                ),
                "mean_target_norm": detail["mean_target_norm"],
                "mean_progress": detail["mean_progress"],
                "mean_response": detail["mean_response"],
                "mean_cosine": detail["mean_cosine"],
                "max_normalized_risk": detail["max_normalized_risk"],
            }
        )

    aggregate = {}
    for label, selected in (
        ("detected", [e for e in episodes if e["detected"]]),
        ("missed", [e for e in episodes if not e["detected"]]),
    ):
        aggregate[label] = {
            "episodes": len(selected),
            "starts": [e["start"] for e in selected],
            "start": summarize([e["start"] for e in selected]),
            "same_intended_steps_after_onset": summarize([e["same_intended_steps_after_onset"] for e in selected]),
            "eef_separation_at_active_end_m": summarize([e["eef_separation_at_active_end_m"] for e in selected]),
            "mean_target_norm": summarize([e["mean_target_norm"] for e in selected]),
            "mean_progress": summarize([e["mean_progress"] for e in selected]),
            "mean_response": summarize([e["mean_response"] for e in selected]),
            "mean_cosine": summarize([e["mean_cosine"] for e in selected]),
            "max_normalized_risk": summarize([e["max_normalized_risk"] for e in selected]),
        }

    report = {
        "paths": {"normal": NORMAL, "disturbed": DISTURBED},
        "frozen_train_loss": loss,
        "normal_evaluation": normal_eval,
        "disturbed_evaluation": disturbed_eval,
        "paired_integrity": {
            "episodes": len(episodes),
            "max_pre_action_abs_difference": max(e["pre_action_max_abs_difference"] for e in episodes),
            "max_pre_eef_after_difference_m": max(e["pre_eef_after_max_difference_m"] for e in episodes),
        },
        "aggregate": aggregate,
        "episodes": episodes,
    }
    pathlib.Path(OUT).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "paired_integrity": report["paired_integrity"],
        "normal_summary": {k: v for k, v in normal_eval.items() if k != "episodes_detail"},
        "disturbed_summary": {k: v for k, v in disturbed_eval.items() if k != "episodes_detail"},
        "aggregate": aggregate,
    }, ensure_ascii=False, indent=2))
    for episode in episodes:
        print("EPISODE", json.dumps(episode, ensure_ascii=False))


if __name__ == "__main__":
    main()
