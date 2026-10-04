#!/usr/bin/env python3
"""Post-hoc, deployable-signal diagnosis of LIBERO-10 transfer errors.

This script is diagnostic only.  It must not be used to retune the frozen
LIBERO-10 result.  Simulator-only contact/object/reward signals are excluded.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


PATTERN = re.compile(
    r"libero_10_task(?P<task>\d+)_seed32_(?P<condition>normal|scale025|scale050|scale075)"
)
EPS = 1e-12
PANDA_Q_LOW = np.asarray([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
PANDA_Q_HIGH = np.asarray([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])


def causal_mean(values, groups, width=3):
    out = np.empty_like(values, dtype=float)
    starts = {}
    for i, group in enumerate(groups):
        start = starts.get(group, i)
        starts[group] = start
        out[i] = np.mean(values[max(start, i - width + 1) : i + 1])
    return out


def safe_cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + EPS))


def percentile_summary(x):
    x = np.asarray(x, dtype=float)
    if not len(x):
        return {"n": 0}
    return {
        "n": int(len(x)),
        "mean": float(np.mean(x)),
        "p10": float(np.percentile(x, 10)),
        "median": float(np.median(x)),
        "p90": float(np.percentile(x, 90)),
    }


def rank_auc(x, y):
    """AUC with average ranks, implemented without sklearn/scipy."""
    x, y = np.asarray(x, float), np.asarray(y, bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if not n1 or not n0:
        return None
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), float)
    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and x[order[j]] == x[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + 1 + j) / 2.0
        i = j
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def load_rows(root):
    rows = []
    paths = sorted(root.glob("*.jsonl"))
    for file_index, path in enumerate(paths):
        match = PATTERN.fullmatch(path.stem)
        if not match:
            continue
        previous = {}
        for line in path.open(encoding="utf-8"):
            row = json.loads(line)
            if row.get("event") != "step":
                continue
            key = (int(row["task_id"]), int(row["episode_idx"]))
            cmd = np.asarray(row["intended_target_translation"], float)
            actual = np.asarray(row["actual_translation"], float)
            action = np.asarray(row["intended_action"], float)
            qvel = np.asarray(row["joint_vel_before"], float)
            qpos = np.asarray(row["joint_pos_before"], float)
            eef_pos = np.asarray(row["eef_pos_before"], float)
            grip = np.asarray(row["gripper_qpos_before"], float)
            grip_vel = np.asarray(row["gripper_qvel_before"], float)
            prev = previous.get(key)
            record = {
                "file_index": file_index,
                "task": key[0],
                "episode": key[1],
                "condition": match["condition"],
                "action_index": int(row["action_index"]),
                "active": bool(row["disturbance_active"]),
                "command_norm": float(np.linalg.norm(cmd)),
                "actual_norm": float(np.linalg.norm(actual)),
                "same_step_progress": float(np.dot(cmd, actual) / (np.dot(cmd, cmd) + EPS)),
                "same_step_cosine": safe_cosine(cmd, actual),
                "rotation_command_norm": float(np.linalg.norm(action[3:6])),
                "gripper_command": float(action[6]),
                "gripper_aperture": float(abs(grip[0] - grip[1])),
                "gripper_speed": float(np.linalg.norm(grip_vel)),
                "joint_speed": float(np.linalg.norm(qvel)),
                "joint_limit_margin": float(np.min(np.minimum(
                    (qpos - PANDA_Q_LOW) / (PANDA_Q_HIGH - PANDA_Q_LOW),
                    (PANDA_Q_HIGH - qpos) / (PANDA_Q_HIGH - PANDA_Q_LOW),
                ))),
                "eef_radius_xy": float(np.linalg.norm(eef_pos[:2])),
                "eef_height": float(eef_pos[2]),
                "chunk_phase": int(row["action_index"]) % 5,
            }
            if prev is None:
                record.update(
                    command_turn_cosine=1.0,
                    command_delta_norm=0.0,
                    joint_accel_proxy=0.0,
                    eef_speed_change=0.0,
                    lag1_progress=record["same_step_progress"],
                )
            else:
                pcmd, pqvel, pactual = prev
                record.update(
                    command_turn_cosine=safe_cosine(pcmd, cmd),
                    command_delta_norm=float(np.linalg.norm(cmd - pcmd)),
                    joint_accel_proxy=float(np.linalg.norm(qvel - pqvel)),
                    eef_speed_change=float(np.linalg.norm(actual) - np.linalg.norm(pactual)),
                    lag1_progress=float(np.dot(pcmd, actual) / (np.dot(pcmd, pcmd) + EPS)),
                )
            previous[key] = (cmd, qvel, actual)
            rows.append(record)
    return rows, paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-root", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    rows, paths = load_rows(args.trace_root)
    score_data = np.load(args.scores)
    lookup = {
        (int(fi), int(ep), int(ai)): float(score)
        for fi, ep, ai, score in zip(
            score_data["file_index"], score_data["episode_idx"],
            score_data["action_index"], score_data["score"]
        )
    }
    all_row_count = len(rows)
    rows = [
        r for r in rows
        if (r["file_index"], r["episode"], r["action_index"]) in lookup
    ]
    groups = [(r["file_index"], r["episode"]) for r in rows]
    raw_scores = np.asarray([
        lookup[(r["file_index"], r["episode"], r["action_index"])] for r in rows
    ])
    smooth = causal_mean(raw_scores, groups, 3)
    for row, score in zip(rows, smooth):
        row["score_mean3"] = float(score)
        row["alarm"] = bool(score > args.threshold)

    features = [
        "command_norm", "actual_norm", "same_step_progress", "same_step_cosine",
        "lag1_progress", "command_turn_cosine", "command_delta_norm",
        "rotation_command_norm", "gripper_command", "gripper_aperture",
        "gripper_speed", "joint_speed", "joint_accel_proxy", "eef_speed_change",
        "joint_limit_margin", "eef_radius_xy", "eef_height", "chunk_phase",
    ]
    groups_named = {
        "normal_true_negative": [r for r in rows if r["condition"] == "normal" and not r["alarm"]],
        "normal_false_alarm": [r for r in rows if r["condition"] == "normal" and r["alarm"]],
        "active_detected": [r for r in rows if r["active"] and r["alarm"]],
        "active_missed": [r for r in rows if r["active"] and not r["alarm"]],
        "active_scale075": [r for r in rows if r["active"] and r["condition"] == "scale075"],
    }
    summaries = {
        name: {feature: percentile_summary([r[feature] for r in subset]) for feature in features}
        for name, subset in groups_named.items()
    }

    normal = [r for r in rows if r["condition"] == "normal"]
    active = [r for r in rows if r["active"]]
    discrimination = {}
    for feature in features:
        auc_fa = rank_auc([r[feature] for r in normal], [r["alarm"] for r in normal])
        auc_det = rank_auc([r[feature] for r in active], [r["alarm"] for r in active])
        discrimination[feature] = {
            "normal_false_alarm_auc": auc_fa,
            "active_detection_auc": auc_det,
            "normal_false_alarm_separation": None if auc_fa is None else max(auc_fa, 1 - auc_fa),
            "active_detection_separation": None if auc_det is None else max(auc_det, 1 - auc_det),
        }

    phase_counts = defaultdict(lambda: {"normal_steps": 0, "normal_alarms": 0, "active_steps": 0, "active_alarms": 0})
    for row in rows:
        p = str(row["chunk_phase"])
        if row["condition"] == "normal":
            phase_counts[p]["normal_steps"] += 1
            phase_counts[p]["normal_alarms"] += int(row["alarm"])
        if row["active"]:
            phase_counts[p]["active_steps"] += 1
            phase_counts[p]["active_alarms"] += int(row["alarm"])
    for value in phase_counts.values():
        value["normal_alarm_rate"] = value["normal_alarms"] / max(value["normal_steps"], 1)
        value["active_recall"] = value["active_alarms"] / max(value["active_steps"], 1)

    false_alarm_runs = []
    normal_episodes = defaultdict(list)
    for row in rows:
        if row["condition"] == "normal":
            normal_episodes[(row["task"], row["episode"], row["file_index"])].append(row)
    for (task, episode, file_index), episode_rows in normal_episodes.items():
        runs = []
        for row in (r for r in episode_rows if r["alarm"]):
            if not runs or row["action_index"] > runs[-1][-1]["action_index"] + 1:
                runs.append([row])
            else:
                runs[-1].append(row)
        for run in runs:
            false_alarm_runs.append({
                "task": task, "episode": episode, "file_index": file_index,
                "start": run[0]["action_index"], "end": run[-1]["action_index"],
                "length": len(run),
                "median_gripper_aperture": float(np.median([r["gripper_aperture"] for r in run])),
                "median_gripper_command": float(np.median([r["gripper_command"] for r in run])),
                "median_progress": float(np.median([r["same_step_progress"] for r in run])),
                "median_turn_cosine": float(np.median([r["command_turn_cosine"] for r in run])),
                "median_joint_speed": float(np.median([r["joint_speed"] for r in run])),
                "chunk_phases": [r["chunk_phase"] for r in run],
            })

    active_by_condition = {}
    for condition in ("scale025", "scale050", "scale075"):
        subset = [r for r in rows if r["condition"] == condition and r["active"]]
        progress = [r["same_step_progress"] for r in subset]
        active_by_condition[condition] = {
            "steps": len(subset), "alarms": sum(r["alarm"] for r in subset),
            "progress_p10": float(np.percentile(progress, 10)),
            "progress_median": float(np.median(progress)),
            "progress_p90": float(np.percentile(progress, 90)),
            "closed_gripper_fraction": float(np.mean([r["gripper_aperture"] < 0.05 for r in subset])),
        }

    result = {
        "status": "post-hoc diagnostic only; LIBERO-10 remains opened and ineligible as a future holdout",
        "deployable_inputs_only": True,
        "excluded": ["reward", "success", "object state", "simulator contact/force oracle", "disturbance flag as an input"],
        "files": len(paths), "raw_steps": all_row_count, "scored_steps": len(rows),
        "unscored_warmup_steps": all_row_count - len(rows), "threshold": args.threshold,
        "group_counts": {k: len(v) for k, v in groups_named.items()},
        "summaries": summaries, "univariate_discrimination": discrimination,
        "chunk_phase": dict(phase_counts), "normal_false_alarm_runs": false_alarm_runs,
        "active_by_condition": active_by_condition,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    ranked_fa = sorted(features, key=lambda f: discrimination[f]["normal_false_alarm_separation"] or 0, reverse=True)
    ranked_det = sorted(features, key=lambda f: discrimination[f]["active_detection_separation"] or 0, reverse=True)
    def med(group, feature):
        return summaries[group][feature].get("median")
    lines = [
        "# LIBERO-10 误报与漏检的机器人学机制诊断",
        "",
        "> 本报告属于打开测试集后的诊断，不可用于修改既有冻结结果，也不可再把 LIBERO-10 当作最终留出集。",
        "",
        "## 样本分组",
        "",
        f"- 正常真阴性：{len(groups_named['normal_true_negative'])}",
        f"- 正常误报：{len(groups_named['normal_false_alarm'])}",
        f"- 扰动期检出：{len(groups_named['active_detected'])}",
        f"- 扰动期漏检：{len(groups_named['active_missed'])}",
        f"- 0.75 倍扰动期：{len(groups_named['active_scale075'])}",
        "",
        "## 最有解释力的单变量（仅诊断相关性）",
        "",
        "| 正常误报排序 | 分离度 | 扰动检出排序 | 分离度 |",
        "|---|---:|---|---:|",
    ]
    for i in range(min(8, len(features))):
        f1, f2 = ranked_fa[i], ranked_det[i]
        lines.append(f"| {f1} | {discrimination[f1]['normal_false_alarm_separation']:.3f} | {f2} | {discrimination[f2]['active_detection_separation']:.3f} |")
    lines += ["", "## 关键中位数对比", "", "| 特征 | 正常真阴性 | 正常误报 | 扰动检出 | 扰动漏检 | 0.75倍扰动 |", "|---|---:|---:|---:|---:|---:|"]
    for feature in ("command_norm", "actual_norm", "same_step_progress", "lag1_progress", "command_turn_cosine", "command_delta_norm", "joint_speed", "joint_accel_proxy", "joint_limit_margin", "eef_radius_xy", "eef_height", "gripper_aperture", "gripper_speed"):
        lines.append("| {} | {} | {} | {} | {} | {} |".format(feature, *[f"{med(g, feature):.6g}" for g in groups_named]))
    lines += [
        "", "## 正常误报的连续事件", "",
        "| task/episode | 步区间 | 长度 | 夹爪开度中位数 | 进展中位数 | 关节速度中位数 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for run in false_alarm_runs:
        lines.append(f"| {run['task']}/{run['episode']} | {run['start']}-{run['end']} | {run['length']} | {run['median_gripper_aperture']:.4f} | {run['median_progress']:.4f} | {run['median_joint_speed']:.4f} |")
    lines += [
        "", "## 各扰动强度的可观测重叠", "",
        "| 条件 | 报警步/扰动步 | progress p10 | 中位数 | p90 | 闭爪比例 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for condition, value in active_by_condition.items():
        lines.append(f"| {condition} | {value['alarms']}/{value['steps']} | {value['progress_p10']:.4f} | {value['progress_median']:.4f} | {value['progress_p90']:.4f} | {value['closed_gripper_fraction']:.3f} |")
    lines += [
        "", "## 解释边界", "",
        "AUC/分离度表示变量与既有报警之间的相关性，不证明因果，也不是可部署阈值。",
        "标签只用于离线分组；候选输入没有使用扰动标志、奖励、成功结果、物体真值或仿真接触真值。",
        "下一阶段应从这些机制构造预注册的序列假设，再在新的、未查看任务或数据集上冻结验证。",
    ]
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"group_counts": result["group_counts"], "top_false_alarm": ranked_fa[:5], "top_detection": ranked_det[:5]}, indent=2))


if __name__ == "__main__":
    main()
