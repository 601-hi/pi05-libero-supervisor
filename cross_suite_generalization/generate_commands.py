#!/usr/bin/env python3
"""Generate (never execute) a staged, cross-suite LIBERO collection manifest."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--matrix", type=Path, default=Path(__file__).with_name("EXPERIMENT_MATRIX.json"))
    p.add_argument("--remote-root", default="/root/gpufree-data/vla-workspace/openpi")
    p.add_argument("--trace-root", default="/root/gpufree-data/libero-traces/cross_suite_v1")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    plan = json.loads(args.matrix.read_text(encoding="utf-8"))
    fixed = plan["fixed"]
    commands = []
    for split, split_spec in plan["splits"].items():
        for suite, task_ids in split_spec["suites"].items():
            for task_id in task_ids:
                for condition in split_spec["conditions"]:
                    scale = float(fixed["conditions"][condition])
                    disturbance_steps = 0 if condition == "normal" else int(fixed["disturbance_steps"])
                    seed = int(split_spec["seed"])
                    stem = f"{split}_{suite}_task{task_id}_seed{seed}_{condition}"
                    trace = f"{args.trace_root}/{stem}.jsonl"
                    visual = f"{args.trace_root}/{stem}_visual"
                    command = (
                        f"cd {args.remote_root} && LIBERO_CONFIG_PATH=/root/gpufree-data/libero-config "
                        f"PYTHONPATH={args.remote_root}/third_party/libero examples/libero/.venv/bin/python "
                        f"examples/libero/monitoring_main.py --args.task-suite-name {suite} "
                        f"--args.task-id {task_id} --args.num-trials-per-task {split_spec['episodes_per_task']} "
                        f"--args.replan-steps {fixed['replan_steps']} --args.seed {seed} "
                        f"--args.sampling-noise-seed {fixed['sampling_noise_seed_base'] + seed} "
                        f"--args.disturbance-start-mode {fixed['disturbance_start_mode']} "
                        f"--args.disturbance-random-start-min {fixed['random_start_range_inclusive'][0]} "
                        f"--args.disturbance-random-start-max {fixed['random_start_range_inclusive'][1]} "
                        f"--args.disturbance-num-steps {disturbance_steps} "
                        f"--args.translation-action-scale {scale} --args.trace-out-path {trace} "
                        f"--args.visual-out-path {visual} --args.visual-stride 1"
                    )
                    commands.append({
                        "split": split, "suite": suite, "task_id": task_id, "condition": condition,
                        "seed": seed, "expected_episodes": split_spec["episodes_per_task"],
                        "trace": trace, "visual": visual,
                        "do_not_read_before_freeze": bool(split_spec.get("sealed_until_freeze", False)),
                        "command": command,
                    })
    manifest = {
        "status": "generated_not_executed",
        "matrix": str(args.matrix),
        "commands": commands,
        "expected_total_episodes": sum(c["expected_episodes"] for c in commands),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"commands": len(commands), "episodes": manifest["expected_total_episodes"]}, indent=2))


if __name__ == "__main__":
    main()
