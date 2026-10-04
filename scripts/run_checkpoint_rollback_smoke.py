"""GPU/MuJoCo plumbing-only replay of bounded historical-corridor rollback.

This is not an evaluation policy and must never be counted as task recovery.
It replays a frozen trace prefix, then asks the deterministic rollback
controller to revisit earlier *actually observed* states while logging every
safety decision.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vla_supervisor.checkpoint_recovery import (
    CartesianRollbackController, HistoricalCorridorPlanner, RecoverySafetyGate,
    WindowedRecoveryResponseMonitor,
)
from vla_supervisor.kinematic_safety import measure_public_kinematic_safety


DUMMY_ACTION = [0.0] * 6 + [-1.0]


def write_row(handle, row):
    handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    handle.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--prefix-actions", type=int, default=30)
    parser.add_argument("--target-action", type=int, default=20)
    parser.add_argument("--waypoint-stride", type=int, default=3)
    parser.add_argument("--steps-per-waypoint", type=int, default=12)
    parser.add_argument("--maximum-settle-steps", type=int, default=20)
    parser.add_argument("--settle-drift-threshold-m", type=float, default=.0003)
    parser.add_argument("--settle-confirmations", type=int, default=3)
    parser.add_argument("--position-tolerance-m", type=float, default=.002)
    args = parser.parse_args()
    if not 0 <= args.target_action < args.prefix_actions:
        raise ValueError("target-action must precede prefix-actions")

    trace_rows = [json.loads(line) for line in args.trace.open(encoding="utf-8") if line.strip()]
    trace_steps = [row for row in trace_rows if row.get("event") == "step"]
    if len(trace_steps) < args.prefix_actions:
        raise ValueError("trace is shorter than requested replay prefix")

    suite = benchmark.get_benchmark_dict()[args.suite]()
    task = suite.get_task(args.task_id)
    initial_state = suite.get_task_init_states(args.task_id)[0]
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=bddl, camera_heights=256, camera_widths=256)
    env.seed(args.seed)
    env.reset()
    obs = env.set_init_state(initial_state)
    for _ in range(10):
        obs, *_ = env.step(DUMMY_ACTION)

    history = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output:
        write_row(output, {
            "event": "rollback_smoke_start", "test_only": True,
            "suite": args.suite, "task_id": args.task_id, "seed": args.seed,
            "prefix_actions": args.prefix_actions, "target_action": args.target_action,
        })
        for position, source in enumerate(trace_steps[:args.prefix_actions]):
            action = np.asarray(source.get("executed_action", source["action"]), dtype=float)
            obs, *_ = env.step(action.tolist())
            measurement = measure_public_kinematic_safety(env.sim, env.robots[0])
            history.append({
                "action_index": position,
                "eef_pos": np.asarray(obs["robot0_eef_pos"], float).tolist(),
                "joint_pos": np.asarray(obs["robot0_joint_pos"], float).tolist(),
                # Test-only replay assumption. It is not a deployable contact-clear label.
                "recovery_safe": True,
            })
            write_row(output, {
                "event": "prefix_replay_step", "action_index": position,
                "eef_pos": history[-1]["eef_pos"], "joint_pos": history[-1]["joint_pos"],
                **measurement,
            })

        waypoints = HistoricalCorridorPlanner(args.waypoint_stride).build(
            history, target_action_index=args.target_action,
            current_action_index=args.prefix_actions)
        controller = CartesianRollbackController()
        gate = RecoverySafetyGate()
        response_monitor = WindowedRecoveryResponseMonitor(gate, window=3)
        reference_action = trace_steps[args.prefix_actions - 1].get(
            "executed_action", trace_steps[args.prefix_actions - 1]["action"])
        hold_action = np.zeros(7, dtype=float)
        hold_action[6] = float(reference_action[6])
        stable_run = 0
        for settle_index in range(args.maximum_settle_steps):
            before = np.asarray(obs["robot0_eef_pos"], float).copy()
            obs, *_ = env.step(hold_action.tolist())
            after = np.asarray(obs["robot0_eef_pos"], float).copy()
            write_row(output, {
                "event": "rollback_settle_step", "settle_step": settle_index + 1,
                "eef_before": before.tolist(), "eef_after": after.tolist(),
                "drift_m": float(np.linalg.norm(after - before)),
            })
            drift = float(np.linalg.norm(after - before))
            stable_run = stable_run + 1 if drift <= args.settle_drift_threshold_m else 0
            if stable_run >= args.settle_confirmations:
                break
        if stable_run < args.settle_confirmations:
            write_row(output, {
                "event": "rollback_smoke_end", "outcome": "settle_failed",
                "rollback_steps": 0, "final_position_error_m": None,
                "strict_safety_certified": False,
                "limitations": ["occupancy_clear_test_assumption", "no_contact_sensor"],
            })
            env.close()
            print(json.dumps({"outcome": "settle_failed", "rollback_steps": 0}, indent=2))
            return
        executed = 0
        outcome = "incomplete"
        for waypoint_number, waypoint in enumerate(waypoints):
            waypoint_reached = False
            for _ in range(args.steps_per_waypoint):
                current_eef = np.asarray(obs["robot0_eef_pos"], float)
                current_joint = np.asarray(obs["robot0_joint_pos"], float)
                target_eef = np.asarray(waypoint.eef_pos, float)
                if np.linalg.norm(target_eef - current_eef) <= args.position_tolerance_m:
                    waypoint_reached = True
                    break
                action = controller.next_action(current_eef, target_eef, reference_action)
                predicted_eef = current_eef + .05 * np.clip(action[:3], -1, 1)
                measurement = measure_public_kinematic_safety(env.sim, env.robots[0])
                corridor_deviation = float(np.max(np.abs(
                    current_joint - np.asarray(waypoint.joint_pos, float))))
                pre = gate.precheck(
                    current_eef=current_eef, predicted_eef=predicted_eef,
                    current_joint=current_joint, predicted_joint=current_joint,
                    joint_margin=(measurement["joint_margin"]
                                  if measurement["joint_margin"] is not None else -1),
                    singularity_sigma=(measurement["singularity_sigma"]
                                       if measurement["singularity_sigma"] is not None else -1),
                    corridor_deviation=corridor_deviation,
                    # The smoke has no calibrated occupancy provider. This test-only
                    # assumption is recorded and cannot authorize deployment.
                    occupancy_clear=True,
                )
                if not pre.safe:
                    outcome = "precheck_rejected"
                    write_row(output, {"event": "rollback_abort", "stage": "precheck",
                                       "reasons": pre.reasons, "waypoint": waypoint.action_index})
                    break
                before = current_eef.copy()
                obs, *_ = env.step(action.tolist())
                after = np.asarray(obs["robot0_eef_pos"], float)
                post = response_monitor.observe(
                    commanded_delta=predicted_eef - before, actual_delta=after - before,
                    contact_probability=0.0, contact_confidence=0.0)
                executed += 1
                write_row(output, {
                    "event": "rollback_step", "rollback_step": executed,
                    "waypoint_number": waypoint_number,
                    "waypoint_action_index": waypoint.action_index,
                    "command": action.tolist(), "eef_before": before.tolist(),
                    "eef_after": after.tolist(), "target_eef": target_eef.tolist(),
                    "precheck": {"state": pre.state, "reasons": pre.reasons},
                    "postcheck": {"state": post.state, "reasons": post.reasons},
                    "test_only_occupancy_assumption": True,
                })
                if post.state not in {"safe", "warming_up"}:
                    outcome = post.state
                    break
            if outcome != "incomplete":
                break
            final_waypoint_error = float(np.linalg.norm(
                np.asarray(obs["robot0_eef_pos"], float) - np.asarray(waypoint.eef_pos, float)))
            if not waypoint_reached and final_waypoint_error > args.position_tolerance_m:
                outcome = "waypoint_timeout"
                write_row(output, {
                    "event": "rollback_abort", "stage": "waypoint_timeout",
                    "waypoint": waypoint.action_index,
                    "waypoint_error_m": final_waypoint_error,
                })
                break
        else:
            target = np.asarray(history[args.target_action]["eef_pos"], float)
            final_error = float(np.linalg.norm(
                np.asarray(obs["robot0_eef_pos"], float) - target))
            outcome = "reached" if final_error <= args.position_tolerance_m else "incomplete"

        target = np.asarray(history[args.target_action]["eef_pos"], float)
        final_error = float(np.linalg.norm(np.asarray(obs["robot0_eef_pos"], float) - target))
        write_row(output, {
            "event": "rollback_smoke_end", "outcome": outcome,
            "rollback_steps": executed, "final_position_error_m": final_error,
            "strict_safety_certified": False,
            "limitations": ["occupancy_clear_test_assumption", "no_contact_sensor"],
        })
    env.close()
    print(json.dumps({"outcome": outcome, "rollback_steps": executed,
                      "final_position_error_m": final_error}, indent=2))


if __name__ == "__main__":
    main()
