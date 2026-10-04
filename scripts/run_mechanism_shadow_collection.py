#!/usr/bin/env python3
"""Run a frozen no-intervention mechanism-shadow collection matrix."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess


def trace_complete(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()]
    except (OSError, json.JSONDecodeError):
        return False
    return sum(row.get("event") == "episode_end" for row in rows) == 1


def artifact_id(item: dict, seed: int) -> str:
    return (f'{item["suite"]}_task{int(item["task_id"])}_seed{seed}_'
            f'noise{int(item["sampling_noise_seed"])}')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("plan", type=Path)
    parser.add_argument("--openpi-root", type=Path,
                        default=Path("/root/gpufree-data/vla-workspace/openpi"))
    parser.add_argument("--output-root", type=Path,
                        default=Path("/root/gpufree-data/libero-traces/mechanism_shadow_v1"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    seed = int(plan["seed"])
    minimum_free = float(plan.get("minimum_free_gib", 6.0)) * 1024 ** 3
    if not args.dry_run and shutil.disk_usage(args.output_root.parent).free < minimum_free:
        raise RuntimeError("insufficient free space for preregistered visual collection")

    monitor = args.openpi_root / "examples/libero/monitoring_main.py"
    python = args.openpi_root / "examples/libero/.venv/bin/python"
    traces = args.output_root / "traces"
    videos = args.output_root / "videos"
    visuals = args.output_root / "visuals"
    for directory in (traces, videos, visuals):
        if not args.dry_run:
            directory.mkdir(parents=True, exist_ok=True)
    frozen = args.output_root / "frozen_plan.json"
    if not args.dry_run:
        if frozen.exists() and frozen.read_bytes() != args.plan.read_bytes():
            raise RuntimeError("output root already belongs to a different frozen plan")
        if not frozen.exists():
            frozen.write_bytes(args.plan.read_bytes())

    env = dict(os.environ)
    env["LIBERO_CONFIG_PATH"] = "/root/gpufree-data/libero-config"
    env["PYTHONPATH"] = str(args.openpi_root / "third_party/libero")
    for ordinal, item in enumerate(plan["episodes"]):
        identity = artifact_id(item, seed)
        trace = traces / f"{identity}.jsonl"
        if trace_complete(trace):
            print("SKIP_COMPLETE", identity, flush=True)
            continue
        if trace.exists():
            raise RuntimeError(f"refusing to overwrite incomplete trace: {trace}")
        command = [
            str(python), str(monitor),
            "--args.task-suite-name", str(item["suite"]),
            "--args.task-id", str(int(item["task_id"])),
            "--args.num-trials-per-task", "1",
            "--args.replan-steps", str(int(plan.get("replan_steps", 5))),
            "--args.seed", str(seed),
            "--args.sampling-noise-seed", str(int(item["sampling_noise_seed"])),
            "--args.trace-out-path", str(trace),
            "--args.video-out-path", str(videos),
            "--args.visual-out-path", str(visuals),
            "--args.visual-stride", str(int(plan.get("visual_stride", 1))),
            "--args.supervisor-enabled",
            "--args.supervisor-execution-mode", "frozen_four_state_visual_background",
        ]
        print(f"RUN {ordinal + 1}/{len(plan['episodes'])}", " ".join(command), flush=True)
        if not args.dry_run:
            if shutil.disk_usage(args.output_root.parent).free < minimum_free:
                raise RuntimeError("free space fell below the frozen safety floor")
            subprocess.run(command, env=env, check=True)


if __name__ == "__main__":
    main()
