import argparse
import collections
import json
import pathlib


parser = argparse.ArgumentParser()
parser.add_argument("trace", type=pathlib.Path)
parser.add_argument("--output", type=pathlib.Path)
args = parser.parse_args()
rows = [json.loads(line) for line in args.trace.open(encoding="utf-8") if line.strip()]
events = collections.Counter(row.get("event") for row in rows)
proposals = [row for row in rows if row.get("event") == "complex_rollback_proposal"]
ends = [row for row in rows if row.get("event") == "episode_end"]

segments = []
for row in proposals:
    key = (row.get("rollback_state"), row.get("target_action_index"))
    if not segments or segments[-1]["key"] != key:
        segments.append({
            "key": key,
            "state": key[0],
            "target_action_index": key[1],
            "first_action_index": row.get("action_index"),
            "last_action_index": row.get("action_index"),
            "proposal_count": 1,
            "first_reached_history_depth": row.get("reached_history_depth"),
            "last_reached_history_depth": row.get("reached_history_depth"),
            "maximum_spatial_retreat_m": row.get("actual_spatial_retreat_m"),
            "selected_candidates": collections.Counter([
                row.get("selected_candidate_id")]),
        })
    else:
        segment = segments[-1]
        segment["last_action_index"] = row.get("action_index")
        segment["proposal_count"] += 1
        segment["last_reached_history_depth"] = row.get("reached_history_depth")
        current_retreat = row.get("actual_spatial_retreat_m")
        if current_retreat is not None:
            previous_retreat = segment.get("maximum_spatial_retreat_m")
            segment["maximum_spatial_retreat_m"] = (
                current_retreat if previous_retreat is None
                else max(previous_retreat, current_retreat))
        segment["selected_candidates"][row.get("selected_candidate_id")] += 1

for segment in segments:
    segment.pop("key")
    segment["selected_candidates"] = dict(segment["selected_candidates"])

report = {
    "event_counts": dict(events),
    "rollback_state_counts": dict(collections.Counter(
        row.get("rollback_state") for row in proposals)),
    "maximum_reached_history_depth": max(
        (row.get("reached_history_depth", 0) for row in proposals), default=0),
    "target_segments": segments,
    "episode_end": ends,
}
payload = json.dumps(report, ensure_ascii=False, indent=2)
print(payload)
if args.output:
    args.output.write_text(payload + "\n", encoding="utf-8")
