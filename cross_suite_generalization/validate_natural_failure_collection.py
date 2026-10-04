#!/usr/bin/env python3
"""Validate stage-A natural collection traces and step-aligned visual sidecars."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--expected-episodes", type=int, default=120)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    trace_root = args.root / "traces"
    visual_root = args.root / "visual_sidecars"
    visual_roots = [visual_root] + sorted(args.root.glob("recovery_*/visual_sidecars"))
    errors, episodes = [], []
    for trace in sorted(trace_root.glob("*.jsonl")):
        grouped = defaultdict(list); inference_rows = defaultdict(list); ends = {}
        for line_number, line in enumerate(trace.open(encoding="utf-8"), 1):
            try:
                row = json.loads(line)
            except Exception as exc:
                errors.append(f"{trace}:{line_number}: invalid JSON: {exc}")
                continue
            if row.get("event") == "step":
                grouped[(int(row["task_id"]), int(row["episode_idx"]))].append(row)
            elif row.get("event") == "inference":
                inference_rows[(int(row["task_id"]), int(row["episode_idx"]))].append(row)
            elif row.get("event") == "episode_end":
                ends[(int(row["task_id"]), int(row["episode_idx"]))] = row
        for key, rows in grouped.items():
            rows.sort(key=lambda x: int(x["action_index"]))
            end = ends.get(key)
            if end is None:
                errors.append(f"{trace}:{key}: missing episode_end")
                continue
            if any(bool(x.get("disturbance_active", False)) for x in rows):
                errors.append(f"{trace}:{key}: disturbance_active in natural collection")
            if int(end.get("disturbed_steps", 0)) != 0:
                errors.append(f"{trace}:{key}: disturbed_steps != 0")
            if bool(end.get("supervisor_enabled", False)):
                errors.append(f"{trace}:{key}: supervisor enabled in natural collection")
            artifact = end.get("artifact_stem")
            outcome_suffix = "success" if bool(end.get("success", False)) else "failure"
            if not artifact:
                # monitoring_main uses a deterministic stem; locate the unique matching episode.
                matches = []
                for root in visual_roots:
                    matches.extend(root.glob(
                        f"*task{key[0]:02d}_episode{key[1]:03d}_*_{outcome_suffix}.npz"
                    ))
            else:
                matches = [visual_root / f"{artifact}.npz"]
            expected = np.asarray([x["action_index"] for x in rows if x.get("visual_frame_index") is not None])
            aligned = []
            for candidate in matches:
                if candidate.is_file():
                    with np.load(candidate) as sidecar:
                        if "action_indices" in sidecar.files and np.array_equal(sidecar["action_indices"], expected):
                            aligned.append(candidate)
            if len(aligned) != 1:
                errors.append(f"{trace}:{key}: expected one action-aligned visual sidecar, got {aligned}")
                continue
            with np.load(aligned[0]) as sidecar:
                required = {"action_indices", "agent_images", "wrist_images"}
                if not required.issubset(sidecar.files):
                    errors.append(f"{aligned[0]}: missing arrays {sorted(required-set(sidecar.files))}")
                    continue
                action_indices = sidecar["action_indices"]
                if not np.array_equal(action_indices, expected):
                    errors.append(f"{aligned[0]}: action_indices do not match JSONL")
                if len(sidecar["agent_images"]) != len(action_indices) or len(sidecar["wrist_images"]) != len(action_indices):
                    errors.append(f"{aligned[0]}: camera array length mismatch")
            episode_inferences = inference_rows.get(key, [])
            hashes = [x.get("sampling_noise_sha256") for x in episode_inferences]
            hashes_complete = bool(hashes) and all(isinstance(value, str) and value for value in hashes)
            expected_inferences = int(end.get("inference_calls", len(episode_inferences)))
            if len(episode_inferences) != expected_inferences:
                errors.append(
                    f"{trace}:{key}: inference row count {len(episode_inferences)} "
                    f"!= episode_end inference_calls {expected_inferences}"
                )
            if end.get("sampling_noise_seed") is not None and not hashes_complete:
                errors.append(f"{trace}:{key}: fixed sampling noise requested but inference hashes incomplete")
            episodes.append({"trace": str(trace), "task_id": key[0], "episode_idx": key[1],
                             "success": bool(end.get("success", False)), "steps": len(rows),
                             "visual_frames": len(expected),
                             "inference_rows": len(episode_inferences),
                             "inference_noise_hashes_present": hashes_complete,
                             "sampling_noise_seed": end.get("sampling_noise_seed"),
                             "supervisor_enabled": bool(end.get("supervisor_enabled", False))})
    summary = {
        "trace_files": len(list(trace_root.glob("*.jsonl"))),
        "episodes": len(episodes),
        "successes": sum(x["success"] for x in episodes),
        "failures": sum(not x["success"] for x in episodes),
        "expected_episodes": args.expected_episodes,
        "episode_count_matches": len(episodes) == args.expected_episodes,
        "errors": len(errors),
        "status": "PASS" if len(episodes) == args.expected_episodes and not errors else "FAIL",
    }
    payload = {"summary": summary, "episodes": episodes, "errors": errors}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
