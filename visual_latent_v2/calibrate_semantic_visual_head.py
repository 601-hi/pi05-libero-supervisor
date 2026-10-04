"""Calibrate one visual head on chunk-level causal transitions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from train_semantic_visual_head import Head


def auc(labels: np.ndarray, score: np.ndarray) -> float:
    labels = labels.astype(bool)
    if labels.sum() == 0 or labels.sum() == len(labels):
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=float)
    ranks[order] = np.arange(1, len(score) + 1)
    # Average ranks for ties.
    for value in np.unique(score):
        mask = score == value
        if mask.sum() > 1:
            ranks[mask] = ranks[mask].mean()
    n_pos, n_neg = labels.sum(), (~labels).sum()
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--base-evaluation", type=Path, required=True)
    parser.add_argument("--max-normal-episode-fp", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    data = np.load(args.dataset, allow_pickle=False)
    cached = np.load(args.features, allow_pickle=False)
    checkpoint = torch.load(args.model, map_location="cpu")
    x = cached["features"].astype(np.float32)
    if not np.array_equal(cached["episode_id"], data["episode_id"]):
        raise ValueError("feature/dataset episode rows do not align")
    model = Head(int(checkpoint["input_dim"]), checkpoint["kind"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    with torch.no_grad():
        probability = torch.sigmoid(model(torch.tensor((x - checkpoint["mean"]) / checkpoint["scale"]))).numpy()

    ambiguous = data["previous_ambiguous_any"].astype(bool)
    active = data["previous_active_any"].astype(bool)
    clean = np.isclose(data["scale"], 1.0)
    eligible = ambiguous
    conditional = eligible & (clean | active)
    conditional_auc = auc(active[conditional], probability[conditional])
    groups = {eid: np.flatnonzero(data["episode_id"] == eid) for eid in np.unique(data["episode_id"])}
    base = json.loads(args.base_evaluation.read_text(encoding="utf-8"))
    base_by_episode = {row["episode_id"]: row for row in base["details"]}

    normal_maxima = []
    for eid, indices in groups.items():
        if clean[indices].all():
            values = probability[indices][eligible[indices]]
            normal_maxima.append(float(values.max()) if len(values) else -np.inf)
    candidates = np.unique(np.r_[normal_maxima, [np.nextafter(x, np.inf) for x in normal_maxima], np.inf])
    choices = []
    for threshold in candidates:
        normal_fp = abnormal_detection = visual_rescues = active_chunk_hits = inactive_chunk_alarms = 0
        for eid, indices in groups.items():
            trigger = eligible[indices] & (probability[indices] >= threshold)
            base_row = base_by_episode[eid]
            if clean[indices].all():
                normal_fp += bool(base_row["any_alarm"] or trigger.any())
            else:
                visual_active = bool(np.any(trigger & active[indices]))
                detected = bool(base_row["detected_during_active"] or visual_active)
                abnormal_detection += detected
                visual_rescues += bool(visual_active and not base_row["detected_during_active"])
                active_chunk_hits += int(np.sum(trigger & active[indices]))
                inactive_chunk_alarms += int(np.sum(trigger & ~active[indices]))
        if normal_fp <= args.max_normal_episode_fp:
            choices.append((abnormal_detection, visual_rescues, active_chunk_hits, -inactive_chunk_alarms, -normal_fp, float(threshold)))
    if not choices:
        raise ValueError("no threshold satisfies normal episode budget")
    best = max(choices)
    report = {
        "model": str(args.model), "threshold": best[-1],
        "max_normal_episode_fp": args.max_normal_episode_fp,
        "normal_episode_false_alarms": -best[-2],
        "abnormal_episode_detections": best[0], "visual_episode_rescues": best[1],
        "active_visual_chunk_hits": best[2], "inactive_visual_chunk_alarms": -best[3],
        "base_abnormal_episode_detections": base["abnormal_episode_detections"],
        "base_normal_episode_false_alarms": base["normal_episode_false_alarms"],
        "conditional_ambiguous_auc": conditional_auc,
        "conditional_rows": int(conditional.sum()), "conditional_active_rows": int(active[conditional].sum()),
        "causal_note": "visual observation at chunk k scores executed chunk k-1",
    }
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
