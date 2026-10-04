import collections
import json
import pathlib
import sys

import numpy as np

TOOLS = pathlib.Path("/root/gpufree-data/supervisor-tools-v1")
sys.path.insert(0, str(TOOLS))
from vla_supervisor.online_rollback import select_rollback_candidate_index


TRACE_DIR = pathlib.Path(
    "/root/gpufree-data/libero-traces/"
    "goal_relation_direct_rollback_v6_cycle_breaker_paired"
)


def load(name):
    with (TRACE_DIR / f"{name}.jsonl").open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def episode_rows(rows, episode_idx, event):
    return [
        row for row in rows
        if row.get("episode_idx") == episode_idx and row.get("event") == event
    ]


def action_vector(row):
    return np.asarray(row.get("executed_action", row.get("action")), dtype=float)


baseline = load("baseline")
control = load("control")
report = {"episodes": []}

for episode_idx in range(5):
    b_steps = episode_rows(baseline, episode_idx, "step")
    c_steps = episode_rows(control, episode_idx, "step")
    b_end = episode_rows(baseline, episode_idx, "episode_end")[0]
    c_end = episode_rows(control, episode_idx, "episode_end")[0]
    proposals = episode_rows(control, episode_idx, "complex_rollback_proposal")
    candidates = episode_rows(control, episode_idx, "complex_rollback_candidate")
    by_action = collections.defaultdict(list)
    for candidate in candidates:
        by_action[candidate["action_index"]].append(candidate)

    first_control = next(
        (index for index, row in enumerate(c_steps)
         if row.get("supervisor_action_source") not in (None, "policy", "policy_chunk")),
        len(c_steps),
    )
    prefix = min(first_control, len(b_steps), len(c_steps))
    max_action_diff = max(
        (float(np.max(np.abs(action_vector(b_steps[i]) - action_vector(c_steps[i]))))
         for i in range(prefix)), default=0.0)
    max_eef_diff = max(
        (float(np.linalg.norm(
            np.asarray(b_steps[i]["eef_pos_after"], dtype=float)
            - np.asarray(c_steps[i]["eef_pos_after"], dtype=float)))
         for i in range(prefix)), default=0.0)

    blocked_proposals = 0
    counterfactual_changes = 0
    effective_changes = []
    transitions = collections.Counter()
    for proposal in proposals:
        transitions[proposal.get("cycle_breaker_transition")] += 1
        blocked = proposal.get("cycle_breaker_blocked_candidate_ids") or []
        if blocked:
            blocked_proposals += 1
        group = by_action[proposal["action_index"]]
        summaries = [{
            "candidate_id": row["candidate_id"],
            "kind": row["candidate_kind"],
            "scale": row["candidate_scale"],
            "clearance_safe": row["clearance_safe"],
            "worst_clearance_gain_m": row["worst_clearance_gain_m"],
            "join_direction_alignment": row["join_direction_alignment"],
            "projected_progress_m": row["projected_progress_m"],
            "baseline_clearance_m": min(
                row["baseline_self_clearance_m"],
                row["baseline_environment_clearance_m"],
            ),
        } for row in group]
        original_index = select_rollback_candidate_index(summaries)
        original_id = None if original_index is None else summaries[original_index]["candidate_id"]
        selected_id = proposal.get("selected_candidate_id")
        if original_id != selected_id:
            counterfactual_changes += 1
            effective_changes.append({
                "action_index": proposal["action_index"],
                "original": original_id,
                "selected": selected_id,
                "blocked": blocked,
            })

    report["episodes"].append({
        "episode_idx": episode_idx,
        "baseline_success": b_end["success"],
        "control_success": c_end["success"],
        "first_control_step": first_control,
        "prefix_steps_compared": prefix,
        "max_prefix_action_difference": max_action_diff,
        "max_prefix_eef_difference_m": max_eef_diff,
        "rollback_final_state": c_end.get("rollback_final_state"),
        "rollback_terminal_reason": c_end.get("rollback_terminal_reason"),
        "rollback_join_steps": c_end.get("rollback_join_steps"),
        "rollback_replay_steps": c_end.get("rollback_replay_steps"),
        "blocked_proposals": blocked_proposals,
        "counterfactual_selection_changes": counterfactual_changes,
        "cycle_transitions": dict(transitions),
        "first_effective_changes": effective_changes[:10],
    })

report["summary"] = {
    "baseline_successes": sum(item["baseline_success"] for item in report["episodes"]),
    "control_successes": sum(item["control_success"] for item in report["episodes"]),
    "all_causal_prefixes_exact": all(
        item["max_prefix_action_difference"] == 0.0
        and item["max_prefix_eef_difference_m"] == 0.0
        for item in report["episodes"]
    ),
    "episodes_with_effective_selection_change": sum(
        item["counterfactual_selection_changes"] > 0 for item in report["episodes"]
    ),
    "total_effective_selection_changes": sum(
        item["counterfactual_selection_changes"] for item in report["episodes"]
    ),
}

output = TOOLS / "outputs" / "cycle_breaker_gpu_v6_audit_20260928.json"
output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(report, ensure_ascii=False, indent=2))
