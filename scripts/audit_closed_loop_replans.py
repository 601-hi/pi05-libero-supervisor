"""Offline causal audit of paired closed-loop replanning episodes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def summarize_window(steps: list[dict]) -> dict:
    if not steps:
        return {"count": 0}
    actual = np.asarray([row["actual_translation"] for row in steps], float)
    intended = np.asarray([row["intended_target_translation"] for row in steps], float)
    path = float(np.linalg.norm(actual, axis=1).sum())
    net = float(np.linalg.norm(np.asarray(steps[-1]["eef_pos_after"], float) -
                               np.asarray(steps[0]["eef_pos_before"], float)))
    command_path = float(np.linalg.norm(intended, axis=1).sum())
    directed = float(np.sum(actual * intended) / (np.sum(intended * intended) + 1e-12))
    return {
        "count": len(steps),
        "eef_path_m": path,
        "eef_net_m": net,
        "net_to_path": net / (path + 1e-12),
        "command_path_m": command_path,
        "directed_progress_fraction": directed,
        "gripper_command_transitions": int(sum(
            np.sign(steps[i]["intended_action"][6]) != np.sign(steps[i - 1]["intended_action"][6])
            for i in range(1, len(steps))
        )),
    }


def action_comparison(baseline: list[dict], control: list[dict], start: int,
                      width: int = 10) -> dict:
    bmap = {int(row["action_index"]): row for row in baseline}
    cmap = {int(row["action_index"]): row for row in control}
    indices = [index for index in range(start, start + width)
               if index in bmap and index in cmap]
    if not indices:
        return {"count": 0}
    ba = np.asarray([bmap[index]["intended_action"] for index in indices], float)
    ca = np.asarray([cmap[index]["intended_action"] for index in indices], float)
    delta = np.linalg.norm(ca - ba, axis=1)
    bt, ct = ba[:, :3], ca[:, :3]
    cosine = np.sum(bt * ct, axis=1) / (
        np.linalg.norm(bt, axis=1) * np.linalg.norm(ct, axis=1) + 1e-12)
    end = indices[-1]
    separation = float(np.linalg.norm(
        np.asarray(cmap[end]["eef_pos_after"], float) -
        np.asarray(bmap[end]["eef_pos_after"], float)))
    return {
        "count": len(indices),
        "action_l2_mean": float(delta.mean()),
        "action_l2_max": float(delta.max()),
        "translation_cosine_median": float(np.median(cosine)),
        "eef_separation_at_window_end_m": separation,
    }


def audit_pair(root: Path, pair: dict) -> dict:
    baseline_path = root / pair["baseline_trace"]
    control_path = root / pair["control_trace"]
    baseline_rows = read_jsonl(baseline_path)
    control_rows = read_jsonl(control_path)
    supervisor_rows = read_jsonl(Path(str(control_path) + ".supervisor.jsonl"))
    baseline = [row for row in baseline_rows if row.get("event") == "step"]
    control = [row for row in control_rows if row.get("event") == "step"]
    baseline_end = [row for row in baseline_rows if row.get("event") == "episode_end"][-1]
    control_end = [row for row in control_rows if row.get("event") == "episode_end"][-1]
    alarms = [row for row in supervisor_rows
              if row.get("event") == "supervisor_decision"
              and row.get("decision", {}).get("request_replan")]
    c_by_index = {int(row["action_index"]): row for row in control}
    events = []
    for ordinal, alarm in enumerate(alarms, start=1):
        alarm_index = int(alarm["action_index"])
        first_controlled = next((
            row for row in control
            if int(row.get("supervisor_control_epoch_before_action", 0)) >= ordinal
            and int(row["action_index"]) > alarm_index
        ), None)
        first_policy = next((
            row for row in control
            if int(row.get("supervisor_control_epoch_before_action", 0)) >= ordinal
            and int(row["action_index"]) > alarm_index
            and row.get("supervisor_action_source") == "policy_chunk"
        ), None)
        before = [c_by_index[index] for index in range(max(0, alarm_index - 9), alarm_index + 1)
                  if index in c_by_index]
        policy_index = None if first_policy is None else int(first_policy["action_index"])
        after = [] if policy_index is None else [
            c_by_index[index] for index in range(policy_index, policy_index + 10)
            if index in c_by_index]
        events.append({
            "ordinal": ordinal,
            "alarm_action_index": alarm_index,
            "reason": alarm["decision"]["reason"],
            "confidence": alarm["decision"]["confidence"],
            "first_controlled_action_index": (
                None if first_controlled is None else int(first_controlled["action_index"])),
            "first_controlled_source": (
                None if first_controlled is None else first_controlled.get("supervisor_action_source")),
            "first_replanned_policy_action_index": policy_index,
            "before_10": summarize_window(before),
            "after_replan_10": summarize_window(after),
            "paired_divergence_10": (
                {"count": 0} if policy_index is None else
                action_comparison(baseline, control, policy_index, 10)),
        })
    return {
        "pair_id": pair["pair_id"],
        "suite": pair.get("suite"),
        "task_id": pair.get("task_id"),
        "baseline_success": bool(baseline_end["success"]),
        "control_success": bool(control_end["success"]),
        "baseline_steps": int(baseline_end["executed_actions"]),
        "control_steps": int(control_end["executed_actions"]),
        "replans": int(control_end.get("supervisor_replans", 0)),
        "alarms": events,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    pairs = [audit_pair(args.manifest.parent, pair) for pair in manifest["pairs"]]
    alarms = [alarm for pair in pairs for alarm in pair["alarms"]]
    report = {
        "schema_version": 1,
        "pair_count": len(pairs),
        "alarm_count": len(alarms),
        "pairs_with_alarms": sum(bool(pair["alarms"]) for pair in pairs),
        "pairs": pairs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps({key: report[key] for key in
                      ("pair_count", "alarm_count", "pairs_with_alarms")}, indent=2))


if __name__ == "__main__":
    main()

