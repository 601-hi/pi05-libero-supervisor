"""Read-only diagnostics for frozen semantic visual cascade v1."""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np
import torch

from train_semantic_visual_head import Head


def probabilities(dataset_path: Path, feature_path: Path, model_path: Path):
    data = np.load(dataset_path, allow_pickle=False)
    cached = np.load(feature_path, allow_pickle=False)
    checkpoint = torch.load(model_path, map_location="cpu")
    if not np.array_equal(data["episode_id"], cached["episode_id"]):
        raise ValueError("dataset/feature row mismatch")
    model = Head(int(checkpoint["input_dim"]), checkpoint["kind"])
    model.load_state_dict(checkpoint["state_dict"]); model.eval()
    x = cached["features"].astype(np.float32)
    with torch.no_grad():
        p = torch.sigmoid(model(torch.tensor((x-checkpoint["mean"])/checkpoint["scale"]))).numpy()
    return data, p


def summarize(values):
    if len(values) == 0:
        return {"count": 0}
    return {"count": int(len(values)), **{
        f"p{q}": float(np.percentile(values, q)) for q in (50, 90, 95, 99, 100)
    }}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    model = root / "candidates/mlp_m1.pt"
    cal, cal_p = probabilities(root/"calibration_semantic_chunks.npz", root/"calibration_head_features.npz", model)
    test, test_p = probabilities(root/"test_seed21_semantic_chunks.npz", root/"test_seed21_head_features.npz", model)
    cal_base = json.loads((root/"calibration_frozen_fusion.json").read_text(encoding="utf-8"))
    test_base = json.loads((root/"test_seed21_base_fusion.json").read_text(encoding="utf-8"))
    cascade = json.loads((root/"test_seed21_frozen_visual_cascade.json").read_text(encoding="utf-8"))
    threshold = float(cascade["threshold"])

    def normal_ids(base):
        return {row["episode_id"] for row in base["details"] if not row["has_active_fault"]}
    cal_normal, test_normal = normal_ids(cal_base), normal_ids(test_base)

    drift = {}
    for task in range(10):
        cal_mask = np.asarray([(eid in cal_normal) for eid in cal["episode_id"]]) & cal["previous_ambiguous_any"] & (cal["task_id"] == task)
        test_mask = np.asarray([(eid in test_normal) for eid in test["episode_id"]]) & test["previous_ambiguous_any"] & (test["task_id"] == task)
        drift[str(task)] = {"calibration_normal": summarize(cal_p[cal_mask]), "test_normal": summarize(test_p[test_mask])}

    detail_by_id = {row["episode_id"]: row for row in cascade["details"]}
    false_chunks = []
    missed = []
    for eid in np.unique(test["episode_id"]):
        indices = np.flatnonzero(test["episode_id"] == eid)
        detail = detail_by_id[eid]
        eligible = test["previous_ambiguous_any"][indices].astype(bool)
        triggers = indices[eligible & (test_p[indices] >= threshold)]
        if not detail["has_active_fault"] and len(triggers):
            for i in triggers:
                phase = int(np.argmax(test["condition_last"][i, 53:58]))
                false_chunks.append({
                    "episode_id": str(eid), "task_id": int(test["task_id"][i]),
                    "scale": float(test["scale"][i]), "chunk_id": int(test["chunk_id"][i]),
                    "observation_action_index": int(test["observation_action_index"][i]),
                    "probability": float(test_p[i]), "chunk_phase": phase,
                    "target_norm_mean_m": float(test["condition_mean"][i, 3]),
                    "target_norm_last_m": float(test["condition_last"][i, 3]),
                    "ambiguous_fraction": float(test["previous_ambiguous_fraction"][i]),
                    "normal_logp_mean": float(test["normal_logp_mean"][i]),
                    "abnormal_logp_mean": float(test["abnormal_logp_mean"][i]),
                })
        if detail["has_active_fault"] and not detail["base_detected"]:
            active_indices = indices[test["previous_active_any"][indices].astype(bool)]
            eligible_active = active_indices[test["previous_ambiguous_any"][active_indices].astype(bool)]
            missed.append({
                "episode_id": str(eid), "task_id": int(test["task_id"][indices[0]]),
                "scale": float(test["scale"][indices[0]]),
                "active_transitions": int(len(active_indices)),
                "eligible_active_transitions": int(len(eligible_active)),
                "max_active_probability": float(test_p[active_indices].max()) if len(active_indices) else None,
                "max_eligible_active_probability": float(test_p[eligible_active].max()) if len(eligible_active) else None,
                "would_cross_frozen_threshold": bool(np.any(test_p[eligible_active] >= threshold)) if len(eligible_active) else False,
            })

    # Multiple scale conditions can be exact paired copies when an episode ends before disturbance onset.
    normal_groups = collections.defaultdict(list)
    for row in cascade["details"]:
        if not row["has_active_fault"]:
            _, task, episode = row["episode_id"].split(":")
            normal_groups[(task, episode)].append(row["episode_id"])
    repeated = {f"{task}:{episode}": values for (task, episode), values in normal_groups.items() if len(values) > 1}

    report = {
        "threshold": threshold,
        "normal_score_drift_by_task": drift,
        "visual_false_trigger_chunks": false_chunks,
        "base_missed_abnormal_episodes": missed,
        "paired_repeated_physical_normal_groups": repeated,
        "counts": {
            "false_trigger_chunks": len(false_chunks),
            "base_missed_abnormal_episodes": len(missed),
            "misses_crossing_visual_threshold": sum(row["would_cross_frozen_threshold"] for row in missed),
            "repeated_normal_groups": len(repeated),
        },
    }
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["counts"], ensure_ascii=False, indent=2))
    print("MISSED")
    print(json.dumps(missed, ensure_ascii=False, indent=2))
    print("FALSE_CHUNKS")
    print(json.dumps(false_chunks, ensure_ascii=False, indent=2))
    print("DRIFT")
    print(json.dumps(drift, ensure_ascii=False, indent=2))
    print("REPEATED_NORMAL")
    print(json.dumps(repeated, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
