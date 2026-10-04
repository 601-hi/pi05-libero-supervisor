#!/usr/bin/env python3
"""Generate, but never execute, the pre-registered GPU collection commands."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, default=Path("ABNORMAL_EXPERT_DATASET_MATRIX.json"))
    parser.add_argument("--remote-root", default="/root/gpufree-data/vla-workspace/openpi")
    parser.add_argument("--trace-root", default="/root/gpufree-data/libero-traces/abnormal_expert_v1")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(); plan = json.loads(args.matrix.read_text(encoding="utf-8"))
    common = plan["common"]; commands = []
    for split, spec in plan["splits"].items():
        seeds = spec.get("environment_seeds", [spec.get("environment_seed")])
        episodes = int(spec.get("episodes_per_task_per_condition_per_seed", spec.get("episodes_per_task_per_condition")))
        for seed in seeds:
            for condition in spec["conditions"]:
                scale = {"normal": 1.0, "scale025": .25, "scale050": .5, "scale075": .75}[condition]
                disturbed_steps = 0 if condition == "normal" else common["disturbance_steps"]
                trace = f"{args.trace_root}/{split}_seed{seed}_{condition}.jsonl"
                command = (
                    f"cd {args.remote_root} && LIBERO_CONFIG_PATH=/root/gpufree-data/libero-config "
                    f"PYTHONPATH={args.remote_root}/third_party/libero examples/libero/.venv/bin/python "
                    f"examples/libero/monitoring_main.py --args.task-suite-name {common['suite']} "
                    f"--args.num-trials-per-task {episodes} --args.replan-steps {common['replan_steps']} "
                    f"--args.seed {seed} --args.sampling-noise-seed {2026090300 + seed} "
                    f"--args.disturbance-start-mode {common['disturbance_start_mode']} "
                    f"--args.disturbance-random-start-min {common['random_start_range_inclusive'][0]} "
                    f"--args.disturbance-random-start-max {common['random_start_range_inclusive'][1]} "
                    f"--args.disturbance-num-steps {disturbed_steps} --args.translation-action-scale {scale} "
                    f"--args.trace-out-path {trace}"
                )
                commands.append({"split": split, "seed": seed, "condition": condition,
                                 "expected_episodes": episodes * len(common["tasks"]), "trace": trace,
                                 "do_not_read_before_freeze": split == "test", "command": command})
    manifest = {"status": "generated_not_executed", "commands": commands,
                "expected_total_episodes": sum(x["expected_episodes"] for x in commands)}
    args.out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"commands": len(commands), "expected_total_episodes": manifest["expected_total_episodes"]}, indent=2))


if __name__ == "__main__": main()
