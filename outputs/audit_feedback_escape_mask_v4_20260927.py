import collections
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from vla_supervisor.online_rollback import EscapeProgressMask


TRACE = pathlib.Path(
    "/root/gpufree-data/libero-traces/goal_relation_direct_rollback_v4_mask_paired/control.jsonl"
)


def main():
    mask = EscapeProgressMask()
    transitions = []
    blocked_after_response = []
    with TRACE.open("r", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("event") != "complex_rollback_response" or row.get("episode_idx") != 2:
                continue
            mask.begin_decision()
            clearance = min(
                row["actual_self_clearance_m"],
                row["actual_environment_clearance_m"],
            )
            previous = mask.last_transition
            mask.observe(row.get("candidate_id"), clearance)
            if mask.last_transition != previous:
                transitions.append({
                    "action_index": row["action_index"],
                    "candidate_id": row.get("candidate_id"),
                    "transition": mask.last_transition,
                    "actual_gain_m": mask.last_actual_gain_m,
                    "multiplier": mask.safety_weight_multiplier,
                    "blocked": sorted(mask.blocked_candidate_ids),
                })
            if row.get("candidate_id") == "escape_axis_0_neg_1":
                blocked_after_response.append({
                    "action_index": row["action_index"],
                    "gain_m": mask.last_actual_gain_m,
                    "blocked": "escape_axis_0_neg_1" in mask.blocked_candidate_ids,
                    "transition": mask.last_transition,
                })

    print("TRANSITION_COUNTS", dict(collections.Counter(
        item["transition"] for item in transitions)))
    print("TRANSITIONS", transitions)
    print("X_NEG_1_RESPONSES", blocked_after_response)


if __name__ == "__main__":
    main()
