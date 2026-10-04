import collections
import json
import pathlib

import numpy as np


HERE = pathlib.Path(__file__).parent
V3 = HERE / "control_task0_seed7_noise2026092700_5ep.jsonl"
V2 = HERE.parent / "goal_relation_direct_rollback_gpu_v2_20260927" / V3.name


def load(path):
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def ordered_unique(values):
    result = []
    for value in values:
        if not result or value != result[-1]:
            result.append(value)
    return result


def main():
    v2, v3 = load(V2), load(V3)
    ends = [row for row in v3 if row.get("event") == "episode_end"]
    print("EPISODE_ENDS")
    for row in ends:
        print({key: row.get(key) for key in (
            "episode_idx", "success", "executed_actions", "rollback_executed_steps",
            "rollback_join_steps", "rollback_replay_steps", "rollback_terminal_reason",
            "policy_budget_reset_count")})

    for episode in range(5):
        old_steps = [row for row in v2 if row.get("event") == "step" and row.get("episode_idx") == episode]
        new_steps = [row for row in v3 if row.get("event") == "step" and row.get("episode_idx") == episode]
        first = next((i for i, row in enumerate(new_steps)
                      if row.get("supervisor_action_source") == "complex_rollback"), len(new_steps))
        n = min(first, len(old_steps), len(new_steps))
        action_diff = max((np.max(np.abs(np.asarray(old_steps[i]["executed_action"])
                                         - np.asarray(new_steps[i]["executed_action"])))
                           for i in range(n)), default=0.0)
        state_diff = max((np.linalg.norm(np.asarray(old_steps[i]["eef_pos_before"])
                                         - np.asarray(new_steps[i]["eef_pos_before"]))
                          for i in range(n)), default=0.0)
        print("PREFIX", episode, "steps", n, "action_max", action_diff,
              "state_max", state_diff)

    print("RECOVERY_SUMMARIES")
    for episode in range(5):
        proposals = [row for row in v3 if row.get("event") == "complex_rollback_proposal"
                     and row.get("episode_idx") == episode]
        responses = [row for row in v3 if row.get("event") == "complex_rollback_response"
                     and row.get("episode_idx") == episode]
        print("EP", episode,
              "targets", ordered_unique((row.get("rollback_state"), row.get("target_action_index"))
                                         for row in proposals),
              "selected", dict(collections.Counter(
                  row.get("selected_candidate_id") for row in proposals)),
              "masked", sum(bool(row.get("escape_progress_mask_active")) for row in proposals))
        if responses:
            clearances = np.asarray([row["actual_self_clearance_m"] for row in responses], float)
            print("  SELF_MM first/min/max/last",
                  *(float(x * 1000) for x in (
                      clearances[0], clearances.min(), clearances.max(), clearances[-1])))

    target62 = [row for row in v3 if row.get("event") == "complex_rollback_response"
                and row.get("episode_idx") == 2 and row.get("action_index", -1) >= 0]
    print("EP2_RESPONSE_CANDIDATES", dict(collections.Counter(
        row.get("candidate_id") for row in target62)))


if __name__ == "__main__":
    main()
