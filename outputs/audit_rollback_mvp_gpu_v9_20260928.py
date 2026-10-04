import argparse
import json
import pathlib

import numpy as np


def load(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def rows_for(rows, episode_idx, event):
    return [row for row in rows
            if row.get("episode_idx") == episode_idx and row.get("event") == event]


def action(row):
    return np.asarray(row.get("executed_action", row.get("action")), dtype=float)


parser = argparse.ArgumentParser()
parser.add_argument("--trace-dir", type=pathlib.Path, required=True)
parser.add_argument("--output", type=pathlib.Path, required=True)
args = parser.parse_args()

strict = load(args.trace_dir / "strict.jsonl")
mvp = load(args.trace_dir / "mvp.jsonl")
report = {"episodes": []}

strict_episode_ids = {
    row["episode_idx"] for row in strict if row.get("event") == "episode_end"}
mvp_episode_ids = {
    row["episode_idx"] for row in mvp if row.get("event") == "episode_end"}
if strict_episode_ids != mvp_episode_ids:
    raise RuntimeError(
        f"episode-end mismatch: strict={sorted(strict_episode_ids)}, "
        f"mvp={sorted(mvp_episode_ids)}")

for episode_idx in sorted(strict_episode_ids):
    s_steps = rows_for(strict, episode_idx, "step")
    m_steps = rows_for(mvp, episode_idx, "step")
    s_end = rows_for(strict, episode_idx, "episode_end")[0]
    m_end = rows_for(mvp, episode_idx, "episode_end")[0]
    novelty = rows_for(mvp, episode_idx, "replan_novelty")
    first_external_action = next(
        (i for i, row in enumerate(m_steps)
         if row.get("supervisor_action_source") not in (None, "policy", "policy_chunk")),
        len(m_steps))
    recovery_proposals = rows_for(mvp, episode_idx, "complex_rollback_proposal")
    intervention_indices = [
        *(row["action_index"] for row in novelty),
        *(row["action_index"] for row in recovery_proposals),
    ]
    first_intervention = min(intervention_indices, default=first_external_action)
    prefix = min(first_intervention, len(s_steps), len(m_steps))
    max_action = max((float(np.max(np.abs(action(s_steps[i]) - action(m_steps[i]))))
                      for i in range(prefix)), default=0.0)
    max_eef = max((float(np.linalg.norm(
        np.asarray(s_steps[i]["eef_pos_after"], dtype=float)
        - np.asarray(m_steps[i]["eef_pos_after"], dtype=float)))
        for i in range(prefix)), default=0.0)
    report["episodes"].append({
        "episode_idx": episode_idx,
        "strict_success": bool(s_end["success"]),
        "mvp_success": bool(m_end["success"]),
        "first_mvp_intervention_step": first_intervention,
        "first_mvp_external_action_step": first_external_action,
        "prefix_steps_compared": prefix,
        "max_prefix_action_difference": max_action,
        "max_prefix_eef_difference_m": max_eef,
        "mvp_final_state": m_end.get("rollback_final_state"),
        "mvp_terminal_reason": m_end.get("rollback_terminal_reason"),
        "mvp_join_steps": m_end.get("rollback_join_steps"),
        "mvp_replay_steps": m_end.get("rollback_replay_steps"),
        "replan_novelty_states": [row.get("state") for row in novelty],
        "replan_rollback_levels": [row.get("rollback_level") for row in novelty],
    })

report["summary"] = {
    "strict_successes": sum(row["strict_success"] for row in report["episodes"]),
    "mvp_successes": sum(row["mvp_success"] for row in report["episodes"]),
    "all_causal_prefixes_exact": all(
        row["max_prefix_action_difference"] == 0.0
        and row["max_prefix_eef_difference_m"] == 0.0
        for row in report["episodes"]),
    "episodes_reaching_replan": sum(
        bool(row["replan_novelty_states"]) for row in report["episodes"]),
    "repeated_replans": sum(
        state == "repeated" for row in report["episodes"]
        for state in row["replan_novelty_states"]),
    "changed_replans": sum(
        state == "changed" for row in report["episodes"]
        for state in row["replan_novelty_states"]),
}

args.output.write_text(
    json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(report, ensure_ascii=False, indent=2))
