"""Audit why mechanism-prompt replanning did or did not recover episodes.

This audit deliberately uses no simulator object state and does not infer a
physical failure mechanism from kinematics alone.  It compares contemporary
paired runs only up to and after the first intervention and emits conservative
failure-response categories for follow-up visual review.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import re

import numpy as np


TASK_RE = re.compile(r"task(\d+)")


def load(path: Path):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [r for r in rows if r.get("event") == "step"], [r for r in rows if r.get("event") == "episode_end"][-1]


def task_id(path: Path) -> int:
    match = TASK_RE.search(path.name)
    if not match:
        raise ValueError(f"cannot parse task id: {path}")
    return int(match.group(1))


def vec(rows, key):
    return np.asarray([r[key] for r in rows], dtype=float)


def path_metrics(rows, start=0, stop=None):
    selected = rows[start:stop]
    if not selected:
        return {"steps": 0, "path_m": 0.0, "net_m": 0.0, "path_to_net": None,
                "radius_m": 0.0, "median_step_m": 0.0}
    before = vec(selected, "eef_pos_before")
    after = vec(selected, "eef_pos_after")
    delta = after - before
    path = float(np.linalg.norm(delta, axis=1).sum())
    net = float(np.linalg.norm(after[-1] - before[0]))
    center = after.mean(axis=0)
    radius = float(np.max(np.linalg.norm(after - center, axis=1)))
    return {
        "steps": len(selected), "path_m": path, "net_m": net,
        "path_to_net": path / max(net, 1e-9), "radius_m": radius,
        "median_step_m": float(np.median(np.linalg.norm(delta, axis=1))),
    }


def replan_indices(rows):
    return [int(r["action_index"]) for r in rows
            if (r.get("supervisor_decision") or {}).get("request_replan")]


def first_diff(a, b, key, start=0, atol=1e-12):
    n = min(len(a), len(b))
    for i in range(start, n):
        if not np.allclose(np.asarray(a[i][key], float), np.asarray(b[i][key], float),
                           rtol=0.0, atol=atol):
            return i
    return None


def paired_after_metrics(plain, prompt, first_replan):
    first_action_diff = first_diff(plain, prompt, "intended_action", first_replan)
    result = {"first_action_difference": first_action_diff}
    if first_action_diff is None:
        return result
    for horizon in (10, 25, 50):
        end = min(len(plain), len(prompt), first_action_diff + horizon)
        if end <= first_action_diff:
            continue
        p0 = np.asarray(prompt[first_action_diff]["eef_pos_before"], float)
        p1 = np.asarray(prompt[end - 1]["eef_pos_after"], float)
        b0 = np.asarray(plain[first_action_diff]["eef_pos_before"], float)
        b1 = np.asarray(plain[end - 1]["eef_pos_after"], float)
        result[f"eef_net_prompt_{horizon}_m"] = float(np.linalg.norm(p1 - p0))
        result[f"eef_net_plain_{horizon}_m"] = float(np.linalg.norm(b1 - b0))
        result[f"eef_separation_{horizon}_m"] = float(np.linalg.norm(p1 - b1))
    return result


def classify(end, replans, rows, paired):
    if end.get("success"):
        return "successful_control"
    if not replans:
        return "unobserved_failure"
    first = replans[0]
    budget = max(1, len(rows) - first)
    safe_stop = (end.get("supervisor_intervention_counts") or {}).get("safe_stop", 0) > 0
    diff = paired.get("first_action_difference")
    sep = paired.get("eef_separation_25_m", paired.get("eef_separation_10_m", 0.0))
    if diff is None:
        return "replan_without_policy_change"
    if safe_stop and budget < 25:
        return "early_safe_stop_after_replan"
    if sep < 0.005:
        return "policy_changed_but_physical_path_nearly_same"
    if len(rows) >= 390:
        return "prompt_diverged_but_timed_out"
    return "prompt_diverged_without_task_recovery"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plain-dir", type=Path, required=True)
    ap.add_argument("--prompt-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    plain_paths = {task_id(p): p for p in args.plain_dir.glob("*.jsonl") if not p.name.endswith(".supervisor.jsonl")}
    prompt_paths = {task_id(p): p for p in args.prompt_dir.glob("*.jsonl") if not p.name.endswith(".supervisor.jsonl") and "repeat" not in p.name}
    records = []
    for tid in sorted(set(plain_paths) & set(prompt_paths)):
        plain, plain_end = load(plain_paths[tid])
        prompt, prompt_end = load(prompt_paths[tid])
        replans = replan_indices(prompt)
        first = replans[0] if replans else None
        paired = paired_after_metrics(plain, prompt, first) if first is not None else {}
        pre_equal = first_diff(plain, prompt, "intended_action", 0, atol=0.0)
        if first is not None:
            pre_equal = pre_equal is None or pre_equal >= first
        else:
            pre_equal = pre_equal is None
        record = {
            "task_id": tid,
            "success": bool(prompt_end.get("success")),
            "steps": len(prompt),
            "plain_success": bool(plain_end.get("success")),
            "plain_steps": len(plain),
            "replan_indices": replans,
            "first_replan_fraction": None if first is None else first / max(1, len(prompt)),
            "pre_intervention_actions_identical": pre_equal,
            "whole_episode": path_metrics(prompt),
            "tail_50": path_metrics(prompt, max(0, len(prompt) - 50)),
            "paired_after_first_replan": paired,
            "safe_stop_count": int((prompt_end.get("supervisor_intervention_counts") or {}).get("safe_stop", 0)),
        }
        record["failure_response_category"] = classify(prompt_end, replans, prompt, paired)
        records.append(record)
    result = {
        "schema_version": 1,
        "evidence_contract": "kinematics and supervisor telemetry only; physical mechanism requires visual review",
        "pair_count": len(records),
        "category_counts": dict(Counter(r["failure_response_category"] for r in records)),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
