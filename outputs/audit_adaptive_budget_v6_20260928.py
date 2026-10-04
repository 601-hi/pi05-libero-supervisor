import json
import pathlib
import sys

sys.path.insert(0, "/root/gpufree-data/supervisor-tools-v1")
from vla_supervisor.adaptive_budget import AdaptivePhaseBudget


TRACE = pathlib.Path(
    "/root/gpufree-data/libero-traces/"
    "goal_relation_direct_rollback_v6_cycle_breaker_paired/control.jsonl"
)
rows = [json.loads(line) for line in TRACE.open(encoding="utf-8") if line.strip()]
result = []

for episode_idx in range(5):
    proposals = {
        row["action_index"]: row for row in rows
        if row.get("event") == "complex_rollback_proposal"
        and row.get("episode_idx") == episode_idx
    }
    responses = [
        row for row in rows
        if row.get("event") == "complex_rollback_response"
        and row.get("episode_idx") == episode_idx
    ]
    governors = {
        "join": AdaptivePhaseBudget(soft_limit=320, hard_limit=640),
        "replay": AdaptivePhaseBudget(soft_limit=160, hard_limit=480),
    }
    counts = {"join": 0, "replay": 0}
    decisions = []
    for response in responses:
        proposal = proposals.get(response["action_index"])
        if proposal is None or proposal.get("rollback_state") not in governors:
            continue
        phase = proposal["rollback_state"]
        counts[phase] += 1
        governors[phase].observe(
            response.get("actual_join_error_m"),
            target_id=proposal.get("target_action_index"),
        )
        if counts[phase] == governors[phase].effective_limit:
            decision = governors[phase].decide(counts[phase])
            decisions.append({
                "phase": phase,
                "steps": counts[phase],
                **decision.__dict__,
            })
    result.append({
        "episode_idx": episode_idx,
        "observed_steps": counts,
        "soft_limit_decisions": decisions,
    })

print(json.dumps(result, ensure_ascii=False, indent=2))
