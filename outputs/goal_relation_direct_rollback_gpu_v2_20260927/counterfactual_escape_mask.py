import collections
import json
import pathlib
import sys


SCRIPT = pathlib.Path(__file__).resolve()
REMOTE_PROJECT = SCRIPT.parents[1]
if (REMOTE_PROJECT / "vla_supervisor").is_dir():
    sys.path.insert(0, str(REMOTE_PROJECT))
else:
    LOCAL_PROJECT = SCRIPT.parents[2]
    sys.path.insert(0, str(LOCAL_PROJECT / "remote_stage_20260927"))

from vla_supervisor.online_rollback import (  # noqa: E402
    EscapeProgressMask,
    select_rollback_candidate_index,
)


TRACE = pathlib.Path(
    "/root/gpufree-data/libero-traces/goal_relation_direct_rollback_v2/"
    "control_task0_seed7_noise2026092700_5ep.jsonl"
)
if not TRACE.exists():
    TRACE = pathlib.Path(__file__).with_name(
        "control_task0_seed7_noise2026092700_5ep.jsonl"
    )


def main():
    groups = collections.defaultdict(list)
    responses = {}
    with TRACE.open("r", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("episode_idx") != 2:
                continue
            if (row.get("event") == "complex_rollback_candidate"
                    and row.get("target_action_index") == 62):
                groups[row["action_index"]].append(row)
            elif row.get("event") == "complex_rollback_response":
                responses[row["action_index"]] = row

    mask = EscapeProgressMask()
    divergences = []
    activation_action = None
    for action_index in sorted(groups):
        rows = groups[action_index]
        summaries = []
        for row in rows:
            action_scale = float(row["candidate_scale"])
            # The recorded action bank scales a common direction.  The exact
            # projected length is immaterial to ranking; scale*alignment is
            # proportional to projected progress within this candidate bank.
            summaries.append({
                "kind": row["candidate_kind"],
                "scale": action_scale,
                "clearance_safe": row["clearance_safe"],
                "worst_clearance_gain_m": row["worst_clearance_gain_m"],
                "join_direction_alignment": row["join_direction_alignment"],
                "projected_progress_m": (
                    action_scale * row["join_direction_alignment"]),
                "baseline_clearance_m": min(
                    row["baseline_self_clearance_m"],
                    row["baseline_environment_clearance_m"]),
            })
        chosen = select_rollback_candidate_index(
            summaries,
            safety_weight_multiplier=mask.safety_weight_multiplier,
        )
        chosen_id = rows[chosen]["candidate_id"] if chosen is not None else None
        recorded_id = next(
            (row["candidate_id"] for row in rows if row.get("selected")), None)
        if chosen_id != recorded_id:
            divergences.append((action_index, recorded_id, chosen_id, mask.active))
        response = responses.get(action_index)
        if response:
            clearance = min(
                response["actual_self_clearance_m"],
                response["actual_environment_clearance_m"],
            )
            was_active = mask.active
            mask.observe(recorded_id, clearance)
            if not was_active and mask.active and activation_action is None:
                activation_action = action_index

    print("MASK_ACTIVATION_ACTION", activation_action)
    print("DIVERGENCE_COUNT", len(divergences))
    print("FIRST_DIVERGENCES", divergences[:10])


if __name__ == "__main__":
    main()
