#!/usr/bin/env python3
"""Validate causal and geometry contracts for one mechanism-shadow rollout."""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open("r", encoding="utf-8") if line.strip()]


def scalar(data, key):
    value = np.asarray(data[key])
    return value.item() if value.ndim == 0 else value.tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--visual", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    trace_rows = rows(args.trace)
    supervisor_rows = rows(Path(str(args.trace) + ".supervisor.jsonl"))
    steps = [row for row in trace_rows if row.get("event") == "step"]
    inferences = [row for row in trace_rows if row.get("event") == "inference"]
    ends = [row for row in trace_rows if row.get("event") == "episode_end"]
    decisions = [row for row in supervisor_rows if row.get("event") == "supervisor_decision"]
    state_counts = collections.Counter()
    source_counts = collections.Counter()
    shadow_actionable = 0
    for decision in decisions:
        for event in decision.get("events", []):
            source_counts[str(event.get("source"))] += 1
            evidence = event.get("evidence") or {}
            if evidence.get("diagnostic_state") is not None:
                state_counts[str(evidence["diagnostic_state"])] += 1
            if evidence.get("shadow_original_event_type") == "object_failure":
                shadow_actionable += 1

    with np.load(args.visual, allow_pickle=False) as data:
        action_indices = np.asarray(data["action_indices"])
        aligned_lengths = {
            key: len(data[key]) for key in (
                "action_indices", "agent_images", "wrist_images",
                "agent_camera_to_world", "wrist_camera_to_world",
                "eef_positions", "eef_quaternions", "gripper_qpos",
            )
        }
        fixed_extrinsic_drift = float(np.max(np.abs(
            np.asarray(data["agent_camera_to_world"])
            - np.asarray(data["agent_camera_to_world"])[0]
        )))
        result = {
            "trace": str(args.trace),
            "visual": str(args.visual),
            "step_count": len(steps),
            "inference_count": len(inferences),
            "episode_end_count": len(ends),
            "success": bool(ends[0]["success"]) if len(ends) == 1 else None,
            "noise_hashes_present": bool(inferences) and all(
                row.get("sampling_noise_sha256") for row in inferences
            ),
            "unique_noise_hashes": len({
                row.get("sampling_noise_sha256") for row in inferences
            }),
            "control_epoch_values": sorted({
                int(row.get("supervisor_control_epoch_before_action", -1)) for row in steps
            }),
            "action_sources": sorted({
                str(row.get("supervisor_action_source")) for row in steps
            }),
            "intervention_enabled": (
                bool(ends[0].get("supervisor_intervention_enabled")) if len(ends) == 1 else None
            ),
            "supervisor_replans": (
                int(ends[0].get("supervisor_replans", -1)) if len(ends) == 1 else None
            ),
            "supervisor_decision_count": len(decisions),
            "diagnostic_states": dict(state_counts),
            "event_sources": dict(source_counts),
            "shadow_actionable_count": shadow_actionable,
            "visual_lengths": aligned_lengths,
            "visual_indices_exact": np.array_equal(action_indices, np.arange(len(steps))),
            "geometry_schema_version": int(scalar(data, "geometry_schema_version")),
            "observation_time": str(scalar(data, "observation_time")),
            "eef_reference": str(scalar(data, "eef_reference")),
            "projection_flip_x": bool(scalar(data, "projection_flip_x")),
            "projection_flip_y": bool(scalar(data, "projection_flip_y")),
            "fixed_extrinsic_max_drift": fixed_extrinsic_drift,
        }

    checks = {
        "one_episode_end": result["episode_end_count"] == 1,
        "fixed_noise_complete": result["noise_hashes_present"],
        "causally_transparent": (
            result["control_epoch_values"] == [0]
            and result["action_sources"] == ["policy_chunk"]
            and result["intervention_enabled"] is False
            and result["supervisor_replans"] == 0
        ),
        "all_visual_arrays_aligned": (
            set(result["visual_lengths"].values()) == {len(steps)}
            and result["visual_indices_exact"]
        ),
        "geometry_contract": (
            result["geometry_schema_version"] == 3
            and result["observation_time"] == "pre_action"
            and result["eef_reference"] == "robot0_grip_site"
            and result["projection_flip_x"] is True
            and result["projection_flip_y"] is False
            and result["fixed_extrinsic_max_drift"] < 1e-12
        ),
        "real_diagnostic_not_placeholder": (
            bool(result["diagnostic_states"])
            and "object_result_placeholder" not in result["event_sources"]
        ),
    }
    result["checks"] = checks
    result["all_checks_pass"] = all(checks.values())
    payload = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    if not result["all_checks_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
