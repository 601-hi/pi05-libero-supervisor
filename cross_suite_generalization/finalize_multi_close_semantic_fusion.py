"""Finalize physical fusion after same-candidate semantic alignment audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from vla_supervisor.multi_event_control_evidence import classify_control_evidence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--physical-fusion", type=Path, required=True)
    parser.add_argument("--semantic-alignment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    physical = json.loads(args.physical_fusion.read_text(encoding="utf-8"))
    alignment = {
        row["event_id"]: row
        for row in json.loads(args.semantic_alignment.read_text(encoding="utf-8"))["records"]
    }
    outputs = []
    for original in physical["records"]:
        row = dict(original)
        semantic = alignment.get(row["event_id"])
        if semantic is not None:
            result = classify_control_evidence(
                wrist_state=row["wrist_state"],
                wrist_passes_gate=bool(row["wrist_passes_frozen_gate"]),
                fixed_has_sustained_motion=bool(row["fixed_sustained_candidate_count"]),
                best_sustained_change=row["best_fixed_sustained_changepoint"],
                same_candidate_semantic_alignment_count=semantic["same_candidate_alignment_count"],
            )
            row.update({
                "presemantic_control_evidence_state": row["control_evidence_state"],
                "same_candidate_semantic_alignment_count": semantic["same_candidate_alignment_count"],
                "control_evidence_state": result.state.value,
                "establish_control": result.establish_control,
                "next_evidence_request": result.next_evidence_request,
            })
        outputs.append(row)
    counts = {}
    for row in outputs:
        counts[row["control_evidence_state"]] = counts.get(row["control_evidence_state"], 0) + 1
    result = {
        "schema_version": 1,
        "safety_statement": "all physical and semantic evidence must align to the same candidate",
        "summary": {"events": len(outputs), "state_counts": counts,
                    "established_control_events": sum(r["establish_control"] for r in outputs)},
        "records": outputs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
