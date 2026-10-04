#!/usr/bin/env python3
"""Audit diagnostic evidence coverage in closed-loop supervisor sidecars."""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path


def audit(root: Path) -> dict:
    sidecars = sorted(root.rglob("*.supervisor.jsonl"))
    event_types = collections.Counter()
    sources = collections.Counter()
    routing_reasons = collections.Counter()
    diagnostic_states = collections.Counter()
    visual_invoked = collections.Counter()
    actionable_diagnostic_events = []
    decision_actions = collections.Counter()

    for path in sidecars:
        for raw in path.open("r", encoding="utf-8"):
            if not raw.strip():
                continue
            row = json.loads(raw)
            if row.get("event") != "supervisor_decision":
                continue
            decision_actions[str(row.get("decision", {}).get("action"))] += 1
            for event in row.get("events", []):
                event_type = str(event.get("event_type"))
                source = str(event.get("source"))
                evidence = event.get("evidence") or {}
                event_types[event_type] += 1
                sources[source] += 1
                if "visual_invoked" in evidence:
                    visual_invoked[str(bool(evidence["visual_invoked"]))] += 1
                if evidence.get("routing_reason") is not None:
                    routing_reasons[str(evidence["routing_reason"])] += 1
                state = evidence.get("diagnostic_state")
                if state is not None:
                    diagnostic_states[str(state)] += 1
                    if event_type == "object_failure":
                        actionable_diagnostic_events.append({
                            "path": str(path),
                            "action_index": row.get("action_index"),
                            "state": str(state),
                            "confidence": event.get("confidence"),
                            "reason": evidence.get("reason"),
                        })

    invoked_total = visual_invoked["True"] + visual_invoked["False"]
    return {
        "root": str(root),
        "sidecar_count": len(sidecars),
        "decision_actions": dict(decision_actions),
        "event_types": dict(event_types),
        "event_sources": dict(sources),
        "visual_invoked": dict(visual_invoked),
        "visual_invocation_rate": (
            visual_invoked["True"] / invoked_total if invoked_total else None
        ),
        "routing_reasons": dict(routing_reasons),
        "diagnostic_states": dict(diagnostic_states),
        "actionable_diagnostic_event_count": len(actionable_diagnostic_events),
        "actionable_diagnostic_events": actionable_diagnostic_events,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = {str(root): audit(root) for root in args.roots}
    payload = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
