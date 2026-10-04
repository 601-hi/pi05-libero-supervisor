#!/usr/bin/env python3
"""Prepare outcome-blind SAM2 inputs from a frozen shadow collection."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open("r", encoding="utf-8") if line.strip()]


def identity(item: dict, seed: int) -> str:
    return (f'{item["suite"]}_task{int(item["task_id"])}_seed{seed}_'
            f'noise{int(item["sampling_noise_seed"])}')


def first_close_frames(step_rows: list[dict], threshold: float = .5) -> list[int]:
    previous = None
    closes = []
    for row in step_rows:
        command = float(row["intended_action"][6])
        sign = 1 if command >= threshold else -1 if command <= -threshold else 0
        if previous == -1 and sign == 1:
            closes.append(int(row["action_index"]))
        if sign in {-1, 1}:
            previous = sign
    return closes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    seed = int(plan["seed"])
    public, private, features = [], [], []
    seen = set()
    for item in plan["episodes"]:
        source_id = identity(item, seed)
        anonymous = "shadow-" + hashlib.sha256(source_id.encode("utf-8")).hexdigest()[:12]
        if anonymous in seen:
            raise RuntimeError("anonymous identifier collision")
        seen.add(anonymous)
        trace = args.root / "traces" / f"{source_id}.jsonl"
        steps = [row for row in rows(trace) if row.get("event") == "step"]
        pattern = (f'rollout_{item["suite"]}_seed{seed}_noise{int(item["sampling_noise_seed"])}_'
                   f'task{int(item["task_id"]):02d}_*.npz')
        matches = list((args.root / "visuals").glob(pattern))
        if len(matches) != 1:
            raise RuntimeError(f"expected one visual sidecar for {source_id}, got {len(matches)}")
        closes = first_close_frames(steps)
        public.append({
            "anonymous_id": anonymous,
            "goal_language": "withheld_not_used_for_physical_candidate_generation",
        })
        private.append({
            "anonymous_id": anonymous,
            "source_identity": source_id,
            "original_trace": str(trace),
            "original_sidecar": str(matches[0]),
        })
        features.append({
            "anonymous_id": anonymous,
            "scored_steps": len(steps),
            "close_frames": closes,
        })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payloads = {
        "public.json": {"schema_version": 1, "records": public},
        "private.json": {"schema_version": 1, "records": private},
        "features.json": {"schema_version": 1, "records": features},
    }
    for name, payload in payloads.items():
        (args.output_dir / name).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps({
        "episodes": len(public),
        "episodes_with_close": sum(bool(row["close_frames"]) for row in features),
        "total_close_transitions": sum(len(row["close_frames"]) for row in features),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
