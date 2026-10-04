"""Audit where frozen four-state execution scoring would request vision.

This is deliberately an alignment/coverage audit.  Episode-level visual
artifacts are never converted into step-level failure labels, and annotation
correctness is not used as a diagnostic signal.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from reliability_sequence_supervisor import ReliabilityCalibration


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--trace-map", type=Path, required=True)
    parser.add_argument("--trace", type=Path, action="append")
    parser.add_argument("--private-map", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--visual-artifact", type=Path, required=True)
    parser.add_argument("--followup-steps", type=int, default=2)
    parser.add_argument("--trigger-after", type=int, default=1)
    parser.add_argument("--manipulation-phase-steps", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    trace_map = json.loads(args.trace_map.read_text(encoding="utf-8"))["files"]
    if args.trace and len(args.trace) != len(trace_map):
        parser.error("repeated --trace order must match trace-map files")
    key_to_file = {(row["suite"], int(row["task_id"])): int(row["file_index"]) for row in trace_map}
    private = json.loads(args.private_map.read_text(encoding="utf-8"))["records"]
    visual = {
        row["anonymous_id"]: row
        for row in json.loads(args.visual_artifact.read_text(encoding="utf-8"))["records"]
    }
    calibration = ReliabilityCalibration.from_json(args.calibration)
    scores = np.load(args.scores, allow_pickle=False)
    phase_active = {}
    if args.manipulation_phase_steps is not None:
        if not args.trace:
            parser.error("--manipulation-phase-steps requires repeated --trace inputs")
        if args.manipulation_phase_steps < 1:
            parser.error("--manipulation-phase-steps must be positive")
        for file_index, path in enumerate(args.trace):
            grouped = {}
            for line in path.open("r", encoding="utf-8"):
                raw = json.loads(line)
                if raw.get("event") != "step":
                    continue
                action = raw.get("intended_action", raw.get("action"))
                grouped.setdefault(int(raw["episode_idx"]), []).append(
                    (int(raw["action_index"]), float(action[6]) if len(action) > 6 else 0.0)
                )
            for episode_index, commands in grouped.items():
                previous_sign = None
                remaining = 0
                for action_index, command in sorted(commands):
                    sign = 1 if command >= 0.5 else -1 if command <= -0.5 else 0
                    transition = previous_sign in {-1, 1} and sign in {-1, 1} and sign != previous_sign
                    if sign in {-1, 1}:
                        previous_sign = sign
                    if transition:
                        remaining = args.manipulation_phase_steps
                    phase_active[(file_index, episode_index, action_index)] = remaining > 0
                    if remaining > 0:
                        remaining -= 1
    state_counter = Counter()
    rows = []
    for episode in private:
        file_index = key_to_file[(episode["suite"], int(episode["task_id"]))]
        mask = ((scores["file_index"] == file_index)
                & (scores["episode_idx"] == int(episode["episode_idx"])))
        indices = np.flatnonzero(mask)
        states = []
        for index in indices:
            result = calibration.score(
                float(scores["normal_logp"][index]),
                float(scores["abnormal_logp"][index]),
                float(scores["ensemble_std"][index]),
                1.0,
            )
            states.append((int(scores["action_index"][index]), result["decision"]))
            state_counter[result["decision"]] += 1

        calls = set()
        ambiguous_run = 0
        keep_alive = 0
        for action_index, state in states:
            if args.manipulation_phase_steps is not None and not phase_active.get(
                (file_index, int(episode["episode_idx"]), action_index), False
            ):
                ambiguous_run = 0
                keep_alive = 0
                continue
            if state == "known_abnormal":
                ambiguous_run = 0
                keep_alive = 0
                continue
            ambiguous = state in {"ambiguous_overlap", "unknown"}
            ambiguous_run = ambiguous_run + 1 if ambiguous else 0
            if ambiguous_run >= args.trigger_after:
                calls.add(action_index)
                keep_alive = args.followup_steps
            elif keep_alive > 0:
                calls.add(action_index)
                keep_alive -= 1
        artifact = visual.get(episode["anonymous_id"])
        rows.append({
            "anonymous_id": episode["anonymous_id"],
            "suite": episode["suite"],
            "task_id": episode["task_id"],
            "episode_idx": episode["episode_idx"],
            "scored_steps": len(states),
            "execution_state_counts": dict(Counter(state for _, state in states)),
            "visual_request_steps": len(calls),
            "visual_request_fraction": len(calls) / len(states) if states else None,
            "episode_level_visual_artifact_available": artifact is not None,
            "visual_artifact_role": "coverage_only_not_step_failure_evidence",
        })

    scored = sum(row["scored_steps"] for row in rows)
    requested = sum(row["visual_request_steps"] for row in rows)
    result = {
        "schema_version": 1,
        "evaluation_role": "retrospective alignment and selective-call coverage only",
        "anti_leakage": (
            "Outcome and identity-correctness annotations are not used to generate visual diagnoses; "
            "episode-level visual outputs are not fabricated into step labels."
        ),
        "summary": {
            "episodes": len(rows),
            "episodes_with_scores": sum(row["scored_steps"] > 0 for row in rows),
            "episodes_with_visual_artifact": sum(row["episode_level_visual_artifact_available"] for row in rows),
            "scored_steps": scored,
            "execution_state_counts": dict(state_counter),
            "selective_visual_request_steps": requested,
            "selective_visual_request_fraction": requested / scored if scored else None,
            "followup_steps": args.followup_steps,
            "trigger_after": args.trigger_after,
            "manipulation_phase_steps": args.manipulation_phase_steps,
        },
        "records": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
