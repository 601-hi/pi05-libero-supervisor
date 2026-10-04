import collections
import json
import pathlib


TRACE = pathlib.Path(
    "/root/gpufree-data/libero-traces/natural_failure_recovery_phase_budget_20260927/"
    "libero90_t0_seed34_natural_failure_dev_control.jsonl"
)


def main():
    rows = [json.loads(line) for line in TRACE.open(encoding="utf-8") if line.strip()]
    proposals = [row for row in rows if row.get("event") == "complex_rollback_proposal"]
    ends = [row for row in rows if row.get("event") == "episode_end"]
    transitions = []
    previous = None
    for row in proposals:
        current = (row.get("rollback_state"), row.get("target_action_index"))
        if current != previous:
            transitions.append((row.get("action_index"), *current))
            previous = current
    print("EVENT_COUNTS:", dict(collections.Counter(row.get("event") for row in rows)))
    print("ROLLBACK_TRANSITIONS:", transitions)
    print("SELECTED_CANDIDATES:", dict(collections.Counter(
        row.get("selected_candidate_id") for row in proposals)))
    print("EPISODE_END:", ends)
    if len(ends) != 1:
        raise SystemExit("expected exactly one episode_end")
    end = ends[0]
    required = {
        "rollback_join_steps", "rollback_replay_steps",
        "rollback_final_state", "rollback_terminal_reason",
    }
    missing = required.difference(end)
    if missing:
        raise SystemExit(f"missing phase audit fields: {sorted(missing)}")


if __name__ == "__main__":
    main()
