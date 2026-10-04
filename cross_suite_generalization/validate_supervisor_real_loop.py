#!/usr/bin/env python3
"""Validate that a supervisor alarm really interrupts and replaces an action chunk."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def max_difference(left: list, right: list) -> float:
    return float(np.max(np.abs(np.asarray(left, dtype=float) - np.asarray(right, dtype=float))))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--supervised", type=Path, required=True)
    parser.add_argument("--supervisor-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    baseline_rows = read_jsonl(args.baseline)
    supervised_rows = read_jsonl(args.supervised)
    supervisor_rows = read_jsonl(args.supervisor_log)
    baseline_steps = [row for row in baseline_rows if row.get("event") == "step"]
    supervised_steps = [row for row in supervised_rows if row.get("event") == "step"]
    baseline_inferences = [row for row in baseline_rows if row.get("event") == "inference"]
    supervised_inferences = [row for row in supervised_rows if row.get("event") == "inference"]
    baseline_end = [row for row in baseline_rows if row.get("event") == "episode_end"][-1]
    supervised_end = [row for row in supervised_rows if row.get("event") == "episode_end"][-1]

    alarm_rows = [
        row for row in supervisor_rows
        if row.get("event") == "supervisor_decision" and row.get("decision", {}).get("request_replan")
    ]
    if len(alarm_rows) != 1:
        raise AssertionError(f"Expected exactly one replan alarm, found {len(alarm_rows)}")
    alarm = alarm_rows[0]
    alarm_index = int(alarm["action_index"])
    pre_alarm_fields = ("intended_action", "executed_action", "eef_pos_before", "eef_pos_after")
    pre_alarm_max_differences = {
        field: max(
            max_difference(baseline_steps[index][field], supervised_steps[index][field])
            for index in range(alarm_index + 1)
        )
        for field in pre_alarm_fields
    }

    interrupted_chunk = baseline_steps[alarm_index]["chunk_id"]
    baseline_chunk_indices = [
        index for index, row in enumerate(baseline_steps) if row["chunk_id"] == interrupted_chunk
    ]
    supervised_chunk_indices = [
        index for index, row in enumerate(supervised_steps) if row["chunk_id"] == interrupted_chunk
    ]
    cleared_action_count = len([index for index in baseline_chunk_indices if index > alarm_index])
    next_index = alarm_index + 1
    post_alarm_action_difference = max_difference(
        baseline_steps[next_index]["intended_action"], supervised_steps[next_index]["intended_action"]
    )

    result = {
        "status": "PASS",
        "claim": "The alarm interrupted the live action chunk and replanned from the current observation.",
        "alarm_action_index": alarm_index,
        "alarm_reason": alarm["decision"]["reason"],
        "decision_rule_evidence_count": len(alarm["decision"]["triggering_events"]),
        "pre_alarm_max_differences": pre_alarm_max_differences,
        "interrupted_chunk_id": interrupted_chunk,
        "baseline_chunk_action_indices": baseline_chunk_indices,
        "supervised_chunk_action_indices": supervised_chunk_indices,
        "cleared_old_actions": cleared_action_count,
        "first_replanned_action_index": next_index,
        "post_alarm_action_difference": post_alarm_action_difference,
        "baseline_inference_steps": [row.get("t") for row in baseline_inferences[:5]],
        "supervised_inference_steps": [row.get("t") for row in supervised_inferences[:5]],
        "baseline_episode_end": baseline_end,
        "supervised_episode_end": supervised_end,
    }

    assert alarm["decision"]["reason"] == "execution_mismatch"
    assert len(alarm["decision"]["triggering_events"]) == 2
    assert max(pre_alarm_max_differences.values()) == 0.0
    assert baseline_chunk_indices == [10, 11, 12, 13, 14]
    assert supervised_chunk_indices == [10, 11]
    assert cleared_action_count == 3
    assert supervised_inferences[3].get("t") == 12
    assert post_alarm_action_difference > 0.0
    assert supervised_end["supervisor_replans"] == 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
