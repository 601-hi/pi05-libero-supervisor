"""Join frozen wrist evidence, fixed motion, and EEF change points by event ID."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from vla_supervisor.multi_event_control_evidence import classify_control_evidence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wrist", type=Path, required=True)
    parser.add_argument("--fixed-motion", type=Path, required=True)
    parser.add_argument("--changepoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    wrist = json.loads(args.wrist.read_text(encoding="utf-8"))["records"]
    fixed = {r["event_id"]: r for r in json.loads(args.fixed_motion.read_text(encoding="utf-8"))["records"]}
    change = {r["event_id"]: r for r in json.loads(args.changepoint.read_text(encoding="utf-8"))["records"]}
    outputs = []
    for row in wrist:
        event_id, fixed_row, change_row = row["event_id"], fixed[row["event_id"]], change[row["event_id"]]
        sustained = set(fixed_row["sustained_candidate_ids"])
        eligible = [r for r in change_row["candidates"] if r["candidate_id"] in sustained]
        best = max((r["combined_change"] for r in eligible), default=None)
        evidence = classify_control_evidence(
            wrist_state=row["state"], wrist_passes_gate=bool(row["passes_frozen_candidate_gate"]),
            fixed_has_sustained_motion=bool(sustained), best_sustained_change=best,
        )
        outputs.append({
            "event_id": event_id, "anonymous_id": row["anonymous_id"],
            "close_ordinal": row["close_ordinal"], "close_frame": row["close_frame"],
            "wrist_state": row["state"], "wrist_passes_frozen_gate": row["passes_frozen_candidate_gate"],
            "fixed_sustained_candidate_count": len(sustained),
            "best_fixed_sustained_changepoint": best,
            "control_evidence_state": evidence.state.value,
            "establish_control": evidence.establish_control,
            "next_evidence_request": evidence.next_evidence_request,
        })
    counts = {}
    for row in outputs:
        counts[row["control_evidence_state"]] = counts.get(row["control_evidence_state"], 0) + 1
    result = {
        "schema_version": 1,
        "safety_statement": "possible_new_control never establishes control without cross-view identity confirmation",
        "summary": {"events": len(outputs), "state_counts": counts,
                    "established_control_events": sum(r["establish_control"] for r in outputs)},
        "records": outputs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
