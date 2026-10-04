import collections
import json
import pathlib

import numpy as np


HERE = pathlib.Path(__file__).parent


def load(name):
    with (HERE / name).open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def ordered_unique(values):
    result = []
    for value in values:
        if not result or result[-1] != value:
            result.append(value)
    return result


def main():
    baseline, control = load("baseline.jsonl"), load("control.jsonl")
    b_ends = {r["episode_idx"]: r for r in baseline if r.get("event") == "episode_end"}
    c_ends = {r["episode_idx"]: r for r in control if r.get("event") == "episode_end"}
    categories = collections.Counter()
    prefix_ok = True

    for episode in range(5):
        b_steps = [r for r in baseline if r.get("event") == "step" and r.get("episode_idx") == episode]
        c_steps = [r for r in control if r.get("event") == "step" and r.get("episode_idx") == episode]
        first = next((i for i, r in enumerate(c_steps)
                      if r.get("supervisor_action_source") == "complex_rollback"), len(c_steps))
        n = min(first, len(b_steps), len(c_steps))
        action_diff = max((np.max(np.abs(np.asarray(b_steps[i]["executed_action"])
                                         - np.asarray(c_steps[i]["executed_action"])))
                           for i in range(n)), default=0.0)
        state_diff = max((np.linalg.norm(np.asarray(b_steps[i]["eef_pos_before"])
                                         - np.asarray(c_steps[i]["eef_pos_before"]))
                          for i in range(n)), default=0.0)
        exact = action_diff == 0.0 and state_diff == 0.0
        prefix_ok &= exact
        bs, cs = bool(b_ends[episode]["success"]), bool(c_ends[episode]["success"])
        category = ("preserved" if bs and cs else "harmed" if bs else
                    "rescued" if cs else "unresolved")
        categories[category] += 1
        proposals = [r for r in control if r.get("event") == "complex_rollback_proposal"
                     and r.get("episode_idx") == episode]
        responses = [r for r in control if r.get("event") == "complex_rollback_response"
                     and r.get("episode_idx") == episode]
        print("EP", episode, "baseline", bs, "control", cs, category,
              "prefix_steps", n, "exact", exact,
              "action_diff", action_diff, "state_diff", state_diff)
        print("  END", {k: c_ends[episode].get(k) for k in (
            "executed_actions", "rollback_executed_steps", "rollback_join_steps",
            "rollback_replay_steps", "rollback_terminal_reason",
            "policy_budget_reset_count")})
        print("  TARGETS", ordered_unique(
            (r.get("rollback_state"), r.get("target_action_index")) for r in proposals))
        print("  SELECTED", dict(collections.Counter(
            r.get("selected_candidate_id") for r in proposals)),
            "MASKED", sum(bool(r.get("escape_progress_mask_active")) for r in proposals))
        if responses:
            values = np.asarray([r["actual_self_clearance_m"] for r in responses], float) * 1000
            print("  SELF_MM", {"first": float(values[0]), "min": float(values.min()),
                                "max": float(values.max()), "last": float(values[-1])})

    print("PREFIX_ALL_EXACT", prefix_ok)
    print("CATEGORIES", dict(categories))


if __name__ == "__main__":
    main()
