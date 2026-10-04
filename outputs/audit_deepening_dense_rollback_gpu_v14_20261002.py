import argparse
import collections
import json
import pathlib


parser = argparse.ArgumentParser()
parser.add_argument("trace", type=pathlib.Path)
parser.add_argument("--output", type=pathlib.Path)
args = parser.parse_args()

rows = [
    json.loads(line)
    for line in args.trace.open(encoding="utf-8")
    if line.strip()
]
events = collections.Counter(row.get("event") for row in rows)
proposals = [row for row in rows if row.get("event") == "complex_rollback_proposal"]
steps = [row for row in rows if row.get("event") == "step"]
ends = [row for row in rows if row.get("event") == "episode_end"]

external = [
    row for row in steps
    if row.get("supervisor_action_source") not in (None, "policy", "policy_chunk")
]
recovery_states = collections.Counter(
    row.get("rollback_state") for row in external
)

attempts = []
for attempt in sorted({
        row.get("recovery_depth_attempt") for row in proposals
        if row.get("recovery_depth_attempt") is not None}):
    attempt_rows = [
        row for row in proposals
        if row.get("recovery_depth_attempt") == attempt
    ]
    attempts.append({
        "attempt": attempt,
        "proposal_count": len(attempt_rows),
        "minimum_replan_steps": sorted({
            row.get("minimum_replan_steps") for row in attempt_rows}),
        "minimum_history_depth": sorted({
            row.get("minimum_replan_history_depth") for row in attempt_rows}),
        "maximum_reached_history_depth": max(
            (row.get("reached_history_depth", 0) for row in attempt_rows),
            default=0),
        "checkpoint_tiers": sorted({
            row.get("checkpoint_tier") for row in attempt_rows
            if row.get("checkpoint_tier") is not None}),
    })

report = {
    "event_counts": dict(events),
    "external_step_count": len(external),
    "external_recovery_states": dict(recovery_states),
    "attempts": attempts,
    "episode_end": ends,
}

payload = json.dumps(report, ensure_ascii=False, indent=2)
print(payload)
if args.output is not None:
    args.output.write_text(payload + "\n", encoding="utf-8")
