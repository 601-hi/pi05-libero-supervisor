import collections
import json
import math
import pathlib


TRACE = pathlib.Path(__file__).with_name(
    "control_task0_seed7_noise2026092700_5ep.jsonl"
)


def load_rows():
    with TRACE.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main():
    rows = load_rows()
    candidates = [
        row
        for row in rows
        if row.get("event") == "complex_rollback_candidate"
        and row.get("episode_idx") == 2
        and row.get("target_action_index") == 62
    ]
    proposals = [
        row
        for row in rows
        if row.get("event") == "complex_rollback_proposal"
        and row.get("episode_idx") == 2
        and row.get("target_action_index") == 62
    ]

    print("TARGET62_PROPOSALS", len(proposals))
    print("TARGET62_CANDIDATES", len(candidates))
    if not proposals:
        return

    first_action = min(row["action_index"] for row in proposals)
    last_action = max(row["action_index"] for row in proposals)
    print("ACTION_RANGE", first_action, last_action)

    selected_counts = collections.Counter(
        row["candidate_id"] for row in candidates if row.get("selected")
    )
    safe_counts = collections.Counter(
        row["candidate_id"] for row in candidates if row.get("clearance_safe")
    )
    print("SELECTED_COUNTS", dict(selected_counts))
    print("SAFE_COUNTS", dict(safe_counts))

    selected_rows = sorted(
        (row for row in candidates if row.get("selected")),
        key=lambda row: row["action_index"],
    )
    print("SELECTED_CLEARANCE_TREND")
    for row in selected_rows[:3] + selected_rows[-3:]:
        print(
            row["action_index"], row["candidate_id"],
            "self=", row["baseline_self_clearance_m"], "->", row["final_self_clearance_m"],
            "env=", row["baseline_environment_clearance_m"], "->", row["final_environment_clearance_m"],
        )

    first = [row for row in candidates if row["action_index"] == first_action]
    print("FIRST_ACTION_CANDIDATES")
    for row in first:
        print(
            row["candidate_id"],
            "safe=", row.get("clearance_safe"),
            "pairwise=", row.get("pairwise_clearance_safe"),
            "selected=", row.get("selected"),
            "self=", row.get("baseline_self_clearance_m"), "->", row.get("final_self_clearance_m"),
            "env=", row.get("baseline_environment_clearance_m"), "->", row.get("final_environment_clearance_m"),
            "gain=", row.get("worst_clearance_gain_m"),
            "align=", row.get("join_direction_alignment"),
        )
        if row["candidate_id"].startswith("direct") and not row.get("pairwise_clearance_safe"):
            before = {(a, b): v for a, b, v in row.get("baseline_environment_pair_clearances", [])}
            after = {(a, b): v for a, b, v in row.get("final_environment_pair_clearances", [])}
            worsened = []
            for pair in sorted(before.keys() | after.keys()):
                b = before.get(pair, math.inf)
                a = after.get(pair, math.inf)
                if a < 0.005 or b < 0.005:
                    worsened.append((pair, b, a, a - b))
            print("  NEAR_PAIR_CHANGES", worsened)

    first_by_id = {row["candidate_id"]: row for row in first}
    print("FIRST_PROPOSAL", [row for row in proposals if row["action_index"] == first_action][0])


if __name__ == "__main__":
    main()
