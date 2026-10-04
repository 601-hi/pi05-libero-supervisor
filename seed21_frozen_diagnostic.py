#!/usr/bin/env python3
"""Post-hoc diagnostic of frozen seed21 two-expert and normal-only results.

This script never tunes a model or threshold. Labels are used only after
causal predictions have been generated, for evaluation and error analysis.
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np
import torch

from normal_dynamics_expert_runtime import _Expert
from two_expert_fusion import TwoExpertFusion


def normal_scores(checkpoint: dict, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    means, variances = [], []
    task = np.argmax(x[:, -10:], axis=1)
    for saved in checkpoint["members"]:
        net = _Expert(int(checkpoint["input_dim"]))
        net.load_state_dict(saved["state_dict"])
        net.eval()
        xm, xs = np.asarray(saved["x_mean"]), np.asarray(saved["x_scale"])
        ym, ys = np.asarray(saved["y_mean"]), np.asarray(saved["y_scale"])
        with torch.no_grad():
            raw = net(torch.tensor((x - xm) / xs, dtype=torch.float32)).numpy()
        means.append(raw[:, :3] * ys + ym)
        variances.append(np.exp(np.clip(raw[:, 3:], -6.0, 3.0)) * ys**2)
    means, variances = np.asarray(means), np.asarray(variances)
    mean = means.mean(axis=0)
    variance = variances.mean(axis=0) + means.var(axis=0)
    scale = np.asarray([checkpoint["variance_scale"][int(t)] for t in task])
    variance = np.maximum(variance * scale, 1e-12)
    innovation = (y - mean) / np.sqrt(variance)
    return np.einsum("ij,ij->i", innovation, innovation)


def persistent_flags(points: list[bool], required: int, window: int) -> np.ndarray:
    history = collections.deque(maxlen=window)
    out = []
    for point in points:
        history.append(bool(point))
        out.append(len(history) >= required and sum(history) >= required)
    return np.asarray(out, dtype=bool)


def percentile(values: np.ndarray, q: float) -> float | None:
    return None if len(values) == 0 else float(np.quantile(values, q))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--scores", type=Path, required=True)
    p.add_argument("--normal", type=Path, required=True)
    p.add_argument("--fusion", type=Path, required=True)
    p.add_argument("--out-json", type=Path, required=True)
    p.add_argument("--out-md", type=Path, required=True)
    args = p.parse_args()

    data = np.load(args.dataset, allow_pickle=False)
    scored = np.load(args.scores, allow_pickle=False)
    if not np.array_equal(data["episode_id"], scored["episode_id"]):
        raise ValueError("dataset and score row order differ")
    checkpoint = torch.load(args.normal, map_location="cpu")
    config = json.loads(args.fusion.read_text(encoding="utf-8"))
    x, y = data["x"].astype(np.float64), data["y"].astype(np.float64)
    nscore = normal_scores(checkpoint, x, y)
    task_ids = data["task_id"].astype(int)
    thresholds = np.asarray([checkpoint["threshold_sets"]["90"][int(t)] for t in task_ids])
    groups: dict[str, list[int]] = collections.defaultdict(list)
    for i, eid in enumerate(data["episode_id"]):
        groups[str(eid)].append(i)

    details = []
    two_step_alarm = np.zeros(len(x), dtype=bool)
    normal_step_alarm = np.zeros(len(x), dtype=bool)
    for eid, indices in groups.items():
        indices.sort(key=lambda i: int(data["action_index"][i]))
        active = data["abnormal"][indices].astype(bool)
        fusion = TwoExpertFusion(config)
        states, ratio_z, novelty_z, point_two = [], [], [], []
        task = int(task_ids[indices[0]])
        c = {**config, **config["by_task"][str(task)]}
        for i in indices:
            nl, al = float(scored["normal_logp"][i]), float(scored["abnormal_logp"][i])
            result = fusion.update(nl, al, task)
            states.append(result.state)
            two_step_alarm[i] = result.persistent_alarm
            point_two.append(result.state == "known_abnormal" or (
                result.state == "unknown_abnormal" and
                (c["normal_logp_center"] - nl) / c["normal_logp_scale"] >= config["unknown_novelty_threshold"]
            ))
            ratio_z.append(((al - nl) - c["ratio_center"]) / c["ratio_scale"])
            novelty_z.append((c["normal_logp_center"] - nl) / c["normal_logp_scale"])
        normal_persistent = persistent_flags(
            list(nscore[indices] > thresholds[indices]),
            int(checkpoint["persistence"]["required"]),
            int(checkpoint["persistence"]["window"]),
        )
        normal_step_alarm[indices] = normal_persistent
        active_idx = np.flatnonzero(active)
        active_global = np.asarray(indices)[active]
        two_detect = bool(two_step_alarm[active_global].any()) if len(active_global) else False
        normal_detect = bool(normal_persistent[active].any()) if active.any() else False
        active_states = collections.Counter(np.asarray(states)[active])
        intended_norm = x[active_global, 3] if len(active_global) else np.asarray([])
        actual_norm = np.linalg.norm(y[active_global], axis=1) if len(active_global) else np.asarray([])
        two_alarm_local = np.flatnonzero(two_step_alarm[indices])
        normal_alarm_local = np.flatnonzero(normal_persistent)
        first_active = int(active_idx[0]) if len(active_idx) else None
        details.append({
            "episode_id": eid,
            "task_id": task,
            "planned_scale": float(data["scale"][indices[0]]),
            "steps": len(indices),
            "active_steps": int(active.sum()),
            "active_start_action": None if not active.any() else int(data["action_index"][active_global[0]]),
            "mean_active_intended_target_norm_m": None if not active.any() else float(intended_norm.mean()),
            "mean_active_actual_translation_norm_m": None if not active.any() else float(actual_norm.mean()),
            "two_expert_detected_active": two_detect,
            "normal_q90_detected_active": normal_detect,
            "two_expert_any_alarm": bool(two_step_alarm[indices].any()),
            "normal_q90_any_alarm": bool(normal_persistent.any()),
            "two_expert_delay_from_active": None if first_active is None or not len(two_alarm_local) else int(two_alarm_local[0] - first_active),
            "normal_q90_delay_from_active": None if first_active is None or not len(normal_alarm_local) else int(normal_alarm_local[0] - first_active),
            "active_state_counts": dict(active_states),
            "active_ratio_z_median": None if not active.any() else float(np.median(np.asarray(ratio_z)[active])),
            "active_novelty_z_median": None if not active.any() else float(np.median(np.asarray(novelty_z)[active])),
            "active_normal_score_ratio_median": None if not active.any() else float(np.median(nscore[active_global] / thresholds[active_global])),
        })

    clean = [d for d in details if d["active_steps"] == 0]
    faults = [d for d in details if d["active_steps"] > 0]
    def metrics(prefix: str) -> dict:
        return {
            "normal_episode_false_alarms": sum(d[f"{prefix}_any_alarm"] for d in clean),
            "normal_episodes": len(clean),
            "abnormal_episode_detections": sum(d[f"{prefix}_detected_active"] for d in faults),
            "abnormal_episodes": len(faults),
        }
    by_scale = {}
    for scale in sorted({d["planned_scale"] for d in details}):
        subset = [d for d in details if d["planned_scale"] == scale]
        exposed = [d for d in subset if d["active_steps"] > 0]
        by_scale[str(scale)] = {
            "planned_episodes": len(subset), "exposed_episodes": len(exposed),
            "zero_exposure_episodes": len(subset) - len(exposed),
            "two_expert_detections": sum(d["two_expert_detected_active"] for d in exposed),
            "normal_q90_detections": sum(d["normal_q90_detected_active"] for d in exposed),
        }
    false_positives = [d for d in clean if d["two_expert_any_alarm"]]
    false_negatives = [d for d in faults if not d["two_expert_detected_active"]]
    true_positives = [d for d in faults if d["two_expert_detected_active"]]
    report = {
        "integrity": {"rows": len(x), "episodes": len(details), "threshold_profile": "90", "labels_used_for_training_or_tuning": False},
        "two_expert": metrics("two_expert"),
        "normal_q90": metrics("normal_q90"),
        "by_scale": by_scale,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "diagnostic_comparison": {
            "fn_mean_active_target_norm_m": None if not false_negatives else float(np.mean([d["mean_active_intended_target_norm_m"] for d in false_negatives])),
            "tp_mean_active_target_norm_m": None if not true_positives else float(np.mean([d["mean_active_intended_target_norm_m"] for d in true_positives])),
            "fn_median_active_steps": percentile(np.asarray([d["active_steps"] for d in false_negatives]), .5),
            "tp_median_active_steps": percentile(np.asarray([d["active_steps"] for d in true_positives]), .5),
        },
        "details": details,
    }
    args.out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# Seed21 冻结监督器误差诊断", "",
        "本报告仅分析冻结测试结果，不使用 seed21 标签训练或调参。", "",
        "## 公平基线", "",
        f"- 双专家：异常检出 {report['two_expert']['abnormal_episode_detections']}/{len(faults)}，正常误报 {report['two_expert']['normal_episode_false_alarms']}/{len(clean)}。",
        f"- 正常专家 q90：异常检出 {report['normal_q90']['abnormal_episode_detections']}/{len(faults)}，正常误报 {report['normal_q90']['normal_episode_false_alarms']}/{len(clean)}。",
        "", "## 各强度的实际暴露与检出", "",
    ]
    for scale, item in by_scale.items():
        lines.append(f"- scale={scale}: 计划 {item['planned_episodes']}，实际暴露 {item['exposed_episodes']}，零暴露 {item['zero_exposure_episodes']}；双专家 {item['two_expert_detections']}，正常 q90 {item['normal_q90_detections']}。")
    lines += ["", "## 双专家误报", "", "```json", json.dumps(false_positives, ensure_ascii=False, indent=2), "```",
              "", "## 双专家漏报", "", "```json", json.dumps(false_negatives, ensure_ascii=False, indent=2), "```", ""]
    args.out_md.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in {"details", "false_positives", "false_negatives"}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
