#!/usr/bin/env python3
"""Audit a frozen mechanism-shadow rollout collection before GPU vision."""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open("r", encoding="utf-8") if line.strip()]


def identity(item: dict, seed: int) -> str:
    return (f'{item["suite"]}_task{int(item["task_id"])}_seed{seed}_'
            f'noise{int(item["sampling_noise_seed"])}')


def visual_pattern(item: dict, seed: int) -> str:
    suite = str(item["suite"])
    task = int(item["task_id"])
    noise = int(item["sampling_noise_seed"])
    return f"rollout_{suite}_seed{seed}_noise{noise}_task{task:02d}_*.npz"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    seed = int(plan["seed"])
    records = []
    aggregate_states = collections.Counter()
    aggregate_routes = collections.Counter()
    aggregate_execution = collections.Counter()

    for item in plan["episodes"]:
        ident = identity(item, seed)
        trace = args.root / "traces" / f"{ident}.jsonl"
        sidecar = Path(str(trace) + ".supervisor.jsonl")
        visual_matches = list((args.root / "visuals").glob(visual_pattern(item, seed)))
        errors = []
        if not trace.is_file():
            errors.append("missing_trace")
        if not sidecar.is_file():
            errors.append("missing_supervisor_sidecar")
        if len(visual_matches) != 1:
            errors.append(f"visual_match_count_{len(visual_matches)}")
        if errors:
            records.append({"identity": ident, "errors": errors})
            continue

        trace_rows = load_rows(trace)
        supervisor_rows = load_rows(sidecar)
        steps = [row for row in trace_rows if row.get("event") == "step"]
        inferences = [row for row in trace_rows if row.get("event") == "inference"]
        ends = [row for row in trace_rows if row.get("event") == "episode_end"]
        decisions = [row for row in supervisor_rows if row.get("event") == "supervisor_decision"]
        states = collections.Counter()
        routes = collections.Counter()
        execution = collections.Counter()
        for decision in decisions:
            for event in decision.get("events", []):
                evidence = event.get("evidence") or {}
                if evidence.get("diagnostic_state") is not None:
                    states[str(evidence["diagnostic_state"])] += 1
                if evidence.get("routing_reason") is not None:
                    routes[str(evidence["routing_reason"])] += 1
                if event.get("source") == "four_state_execution_consistency":
                    execution[str(evidence.get("decision"))] += 1
        aggregate_states.update(states)
        aggregate_routes.update(routes)
        aggregate_execution.update(execution)

        visual = visual_matches[0]
        with np.load(visual, allow_pickle=False) as data:
            lengths = [len(data[key]) for key in (
                "action_indices", "agent_images", "wrist_images",
                "agent_camera_to_world", "wrist_camera_to_world",
                "eef_positions", "eef_quaternions", "gripper_qpos",
            )]
            schema = int(np.asarray(data["geometry_schema_version"]).item())
            indices_exact = np.array_equal(
                np.asarray(data["action_indices"]), np.arange(len(steps))
            )
            fixed_drift = float(np.max(np.abs(
                np.asarray(data["agent_camera_to_world"])
                - np.asarray(data["agent_camera_to_world"])[0]
            )))

        transparent = (
            {int(row.get("supervisor_control_epoch_before_action", -1)) for row in steps} == {0}
            and {str(row.get("supervisor_action_source")) for row in steps} == {"policy_chunk"}
            and len(ends) == 1
            and ends[0].get("supervisor_intervention_enabled") is False
            and int(ends[0].get("supervisor_replans", -1)) == 0
        )
        checks = {
            "one_episode_end": len(ends) == 1,
            "fixed_noise_complete": bool(inferences) and all(
                row.get("sampling_noise_sha256") for row in inferences
            ),
            "causally_transparent": transparent,
            "decision_step_alignment": len(decisions) == len(steps),
            "visual_alignment": set(lengths) == {len(steps)} and indices_exact,
            "geometry_schema": schema == 3 and fixed_drift < 1e-12,
            "real_diagnostic_state": bool(states),
        }
        records.append({
            "identity": ident,
            "suite": item["suite"],
            "task_id": int(item["task_id"]),
            "success": bool(ends[0]["success"]) if len(ends) == 1 else None,
            "steps": len(steps),
            "inferences": len(inferences),
            "diagnostic_states": dict(states),
            "routing_reasons": dict(routes),
            "execution_states": dict(execution),
            "visual_bytes": visual.stat().st_size,
            "checks": checks,
            "all_checks_pass": all(checks.values()),
        })

    complete = [row for row in records if "all_checks_pass" in row]
    suite_results = {}
    for suite in sorted({str(row.get("suite")) for row in complete}):
        selected = [row for row in complete if row["suite"] == suite]
        suite_results[suite] = {
            "episodes": len(selected),
            "successes": sum(bool(row["success"]) for row in selected),
        }
    result = {
        "schema_version": 1,
        "protocol": "frozen_no_intervention_mechanism_shadow",
        "planned_episodes": len(plan["episodes"]),
        "audited_episodes": len(complete),
        "all_episodes_pass": (
            len(complete) == len(plan["episodes"])
            and all(row["all_checks_pass"] for row in complete)
        ),
        "successes": sum(bool(row["success"]) for row in complete),
        "total_steps": sum(row["steps"] for row in complete),
        "total_inferences": sum(row["inferences"] for row in complete),
        "visual_bytes": sum(row["visual_bytes"] for row in complete),
        "suite_results": suite_results,
        "diagnostic_states": dict(aggregate_states),
        "routing_reasons": dict(aggregate_routes),
        "execution_states": dict(aggregate_execution),
        "records": records,
    }
    payload = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in (
        "planned_episodes", "audited_episodes", "all_episodes_pass", "successes",
        "total_steps", "total_inferences", "visual_bytes", "suite_results",
        "diagnostic_states", "routing_reasons", "execution_states",
    )}, ensure_ascii=False, indent=2))
    if not result["all_episodes_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
