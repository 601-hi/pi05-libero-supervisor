"""Run preregistered, causally paired baseline/control LIBERO episodes.

The pi0.5 websocket server must already be listening.  This runner never
selects tasks from observed outcomes and writes the evaluator manifest only
from the frozen plan.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def _run(command: list[str], env: dict[str, str], dry_run: bool) -> None:
    print("RUN", " ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, env=env, check=True)


def _trace_complete(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        return sum(
            json.loads(line).get("event") == "episode_end"
            for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
        ) == 1
    except (OSError, json.JSONDecodeError):
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("plan", type=Path)
    parser.add_argument("--openpi-root", type=Path,
                        default=Path("/root/gpufree-data/vla-workspace/openpi"))
    parser.add_argument("--output-root", type=Path,
                        default=Path("/root/gpufree-data/libero-traces/closed_loop_v1"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    monitor = args.openpi_root / "examples/libero/monitoring_main.py"
    python = args.openpi_root / "examples/libero/.venv/bin/python"
    args.output_root.mkdir(parents=True, exist_ok=True)
    for arm in ("baseline", "control"):
        (args.output_root / "videos" / arm).mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["LIBERO_CONFIG_PATH"] = "/root/gpufree-data/libero-config"
    env["PYTHONPATH"] = str(args.openpi_root / "third_party/libero")
    pairs = []
    for item in plan["episodes"]:
        suite = str(item["suite"])
        task = int(item["task_id"])
        seed = int(item["seed"])
        noise = int(item["sampling_noise_seed"])
        pair_id = str(item.get("pair_id", f"{suite}_task{task}_seed{seed}_noise{noise}"))
        traces = {
            arm: args.output_root / f"{pair_id}_{arm}.jsonl"
            for arm in ("baseline", "control")
        }
        pairs.append({
            "pair_id": pair_id,
            "suite": suite,
            "task_id": task,
            "seed": seed,
            "sampling_noise_seed": noise,
            "baseline_trace": traces["baseline"].name,
            "control_trace": traces["control"].name,
        })
    manifest = {
        "schema_version": 1,
        "protocol": "paired_fixed_initial_state_and_policy_noise",
        "frozen_plan": str(args.plan.resolve()),
        "control": {
            "execution_mode": "frozen_four_state",
            "act_on_ambiguity": False,
            "recovery_prompt": False,
            "stall_retreat": False,
        },
        "pairs": pairs,
    }
    manifest_path = args.output_root / "paired_manifest.json"
    if not args.dry_run:
        # Freeze identities before the first rollout so interruption cannot turn
        # task selection into outcome-dependent selection.
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                                 encoding="utf-8")

    for item, pair in zip(plan["episodes"], pairs):
        suite = str(pair["suite"])
        task = int(pair["task_id"])
        seed = int(pair["seed"])
        noise = int(pair["sampling_noise_seed"])
        pair_id = str(pair["pair_id"])
        traces = {
            "baseline": args.output_root / pair["baseline_trace"],
            "control": args.output_root / pair["control_trace"],
        }
        for arm in ("baseline", "control"):
            trace = traces[arm]
            if not args.overwrite and _trace_complete(trace):
                print("SKIP_COMPLETE", trace, flush=True)
                continue
            command = [
                str(python), str(monitor),
                "--args.task-suite-name", suite,
                "--args.task-id", str(task),
                "--args.num-trials-per-task", "1",
                "--args.replan-steps", str(int(plan.get("replan_steps", 5))),
                "--args.seed", str(seed),
                "--args.sampling-noise-seed", str(noise),
                "--args.trace-out-path", str(trace),
                "--args.video-out-path", str(args.output_root / "videos" / arm),
            ]
            if arm == "control":
                command += [
                    "--args.supervisor-enabled",
                    "--args.supervisor-intervention-enabled",
                    "--args.supervisor-execution-mode", "frozen_four_state",
                ]
            _run(command, env, args.dry_run)
    if args.dry_run:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
