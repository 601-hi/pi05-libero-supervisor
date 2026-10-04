"""Evaluate a pre-frozen visual cascade once on an untouched test split."""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np
import torch

from calibrate_semantic_visual_head import auc
from train_semantic_visual_head import Head


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--base-evaluation", type=Path, required=True)
    parser.add_argument("--frozen-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    data = np.load(args.dataset, allow_pickle=False)
    cached = np.load(args.features, allow_pickle=False)
    checkpoint = torch.load(args.model, map_location="cpu")
    manifest = json.loads(args.frozen_manifest.read_text(encoding="utf-8"))
    threshold = float(manifest["visual_model"]["threshold"])
    if not np.array_equal(cached["episode_id"], data["episode_id"]):
        raise ValueError("feature/dataset row mismatch")
    model = Head(int(checkpoint["input_dim"]), checkpoint["kind"])
    model.load_state_dict(checkpoint["state_dict"]); model.eval()
    x = cached["features"].astype(np.float32)
    with torch.no_grad():
        probability = torch.sigmoid(model(torch.tensor((x-checkpoint["mean"])/checkpoint["scale"]))).numpy()

    eligible = data["previous_ambiguous_any"].astype(bool)
    active = data["previous_active_any"].astype(bool)
    trigger = eligible & (probability >= threshold)
    groups = {eid: np.flatnonzero(data["episode_id"] == eid) for eid in np.unique(data["episode_id"])}
    base = json.loads(args.base_evaluation.read_text(encoding="utf-8"))
    base_by_episode = {row["episode_id"]: row for row in base["details"]}
    details = []
    for eid, indices in groups.items():
        base_row = base_by_episode[eid]
        has_active = bool(base_row["has_active_fault"])
        visual_active = bool(np.any(trigger[indices] & active[indices]))
        visual_any = bool(trigger[indices].any())
        details.append({
            "episode_id": str(eid), "task_id": int(data["task_id"][indices[0]]),
            "scale": float(data["scale"][indices[0]]), "has_active_fault": has_active,
            "base_detected": bool(base_row["detected_during_active"]),
            "base_any_alarm": bool(base_row["any_alarm"]),
            "visual_active_trigger": visual_active, "visual_any_trigger": visual_any,
            "cascade_detected": bool(base_row["detected_during_active"] or visual_active),
            "cascade_any_alarm": bool(base_row["any_alarm"] or visual_any),
            "visual_trigger_chunks": int(trigger[indices].sum()),
            "visual_active_trigger_chunks": int(np.sum(trigger[indices] & active[indices])),
            "visual_inactive_trigger_chunks": int(np.sum(trigger[indices] & ~active[indices])),
        })

    normal = [row for row in details if not row["has_active_fault"]]
    abnormal = [row for row in details if row["has_active_fault"]]
    by_scale = collections.defaultdict(lambda: {"episodes": 0, "base_detected": 0, "cascade_detected": 0})
    for row in abnormal:
        key = str(row["scale"])
        by_scale[key]["episodes"] += 1
        by_scale[key]["base_detected"] += int(row["base_detected"])
        by_scale[key]["cascade_detected"] += int(row["cascade_detected"])

    physical_normal_ids = {row["episode_id"] for row in normal}
    conditional = eligible & np.asarray([
        (str(eid) in physical_normal_ids) or bool(label)
        for eid, label in zip(data["episode_id"], active)
    ])
    report = {
        "threshold": threshold,
        "episodes": len(details), "normal_episodes": len(normal), "abnormal_episodes": len(abnormal),
        "base_normal_episode_false_alarms": sum(row["base_any_alarm"] for row in normal),
        "cascade_normal_episode_false_alarms": sum(row["cascade_any_alarm"] for row in normal),
        "base_abnormal_episode_detections": sum(row["base_detected"] for row in abnormal),
        "cascade_abnormal_episode_detections": sum(row["cascade_detected"] for row in abnormal),
        "visual_episode_rescues": sum(row["cascade_detected"] and not row["base_detected"] for row in abnormal),
        "visual_active_trigger_chunks": sum(row["visual_active_trigger_chunks"] for row in abnormal),
        "visual_inactive_trigger_chunks": sum(row["visual_inactive_trigger_chunks"] for row in details),
        "conditional_ambiguous_auc": auc(active[conditional], probability[conditional]),
        "conditional_rows": int(conditional.sum()), "conditional_active_rows": int(active[conditional].sum()),
        "by_scale": dict(by_scale), "details": details,
        "frozen_manifest": str(args.frozen_manifest),
        "causal_note": "visual trigger at chunk k is credited only to an active executed chunk k-1",
    }
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "details"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
