"""Audit whether traces can support causal trusted-checkpoint reconstruction."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path


REQUIRED_STATE_FIELDS = {
    "eef_pos_after", "joint_pos_after", "joint_vel_after", "action_index",
}
REQUIRED_TRUST_FIELDS = {
    "joint_margin", "singularity_sigma", "contact_clear",
    "recovery_observability", "task_phase",
}


def audit(trace_path: Path) -> dict:
    rows = [json.loads(line) for line in trace_path.open(encoding="utf-8") if line.strip()]
    steps = [row for row in rows if row.get("event") == "step"]
    event_counts = Counter()
    for row in steps:
        for event in (row.get("supervisor_decision") or {}).get("triggering_events", []):
            event_counts[str(event.get("event_type"))] += 1
    state_complete = sum(REQUIRED_STATE_FIELDS.issubset(row) for row in steps)
    trust_presence = {
        field: sum(field in row and row[field] is not None for row in steps)
        for field in sorted(REQUIRED_TRUST_FIELDS)
    }
    return {
        "trace": str(trace_path),
        "steps": len(steps),
        "state_complete_steps": state_complete,
        "state_complete_fraction": state_complete / len(steps) if steps else 0.0,
        "trust_field_present_steps": trust_presence,
        "strict_checkpoint_reconstruction_possible": bool(steps) and all(
            count == len(steps) for count in trust_presence.values()),
        "triggering_event_counts": dict(event_counts),
        "warning": (
            "Do not infer contact clearance or visual observability from absence of an alarm."
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("traces", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    reports = [audit(path) for path in args.traces]
    payload = {
        "schema_version": 1,
        "trace_count": len(reports),
        "strict_ready_count": sum(x["strict_checkpoint_reconstruction_possible"] for x in reports),
        "reports": reports,
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
