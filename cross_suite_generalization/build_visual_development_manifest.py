"""Build an episode-grouped visual development manifest without label-based window selection."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

import numpy as np


TRACE_RE = re.compile(r"(libero_.+)_task(\d+)_seed")
SIDECAR_RE = re.compile(
    r"rollout_(?:(libero_.+?)_)?seed\d+_noise\d+_task(\d+)_episode(\d+)_.*_(success|failure)\.npz"
)


def grouped_split(suite: str, task: int, seed: str) -> str:
    digest = hashlib.sha256(f"{seed}|{suite}|{task}".encode()).digest()
    return "validation" if int.from_bytes(digest[:4], "big") % 5 == 0 else "development"


def response_windows(steps: list[dict], length: int = 5) -> list[dict]:
    records = []
    for start in range(0, len(steps) - length + 1):
        window = steps[start : start + length]
        target = np.asarray([row["intended_target_translation"] for row in window], float)
        actual = np.asarray([row["actual_translation"] for row in window], float)
        target_path = float(np.linalg.norm(target, axis=1).sum())
        if target_path < 0.02:
            continue
        actual_path = float(np.linalg.norm(actual, axis=1).sum())
        records.append({
            "start_action_index": int(window[0]["action_index"]),
            "end_action_index": int(window[-1]["action_index"]),
            "target_path_m": target_path,
            "actual_path_m": actual_path,
            "response_ratio": actual_path / target_path,
        })
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split-seed", default="visual-dev-v1-20260918")
    parser.add_argument("--force-split", default=None)
    args = parser.parse_args()
    traces = args.dataset_root / "traces"
    sidecars = args.dataset_root / "visual_sidecars"

    sidecar_records = []
    for path in sorted(sidecars.glob("*.npz")):
        match = SIDECAR_RE.fullmatch(path.name)
        if match is None:
            continue
        with np.load(path, mmap_mode="r") as archive:
            steps = int(len(archive["action_indices"]))
        sidecar_records.append({
            "path": str(path.resolve()),
            "suite": match.group(1),
            "task_id": int(match.group(2)),
            "episode_idx": int(match.group(3)),
            "success": match.group(4) == "success",
            "steps": steps,
        })

    episodes = []
    for trace in sorted(traces.glob("*.jsonl")):
        match = TRACE_RE.match(trace.stem)
        if match is None:
            continue
        suite, task_id = match.group(1), int(match.group(2))
        rows = [json.loads(line) for line in trace.open(encoding="utf-8") if line.strip()]
        starts = {int(row["episode_idx"]): row for row in rows if row.get("event") == "episode_start"}
        ends = {int(row["episode_idx"]): row for row in rows if row.get("event") == "episode_end"}
        by_episode: dict[int, list[dict]] = {}
        for row in rows:
            if row.get("event") == "step":
                by_episode.setdefault(int(row["episode_idx"]), []).append(row)
        for episode_idx, end in sorted(ends.items()):
            steps = by_episode[episode_idx]
            success = bool(end["success"])
            matches = [record for record in sidecar_records if (
                (record["suite"] is None or record["suite"] == suite)
                and record["task_id"] == task_id
                and record["episode_idx"] == episode_idx
                and record["success"] == success
                and record["steps"] == len(steps)
            )]
            windows = response_windows(steps)
            lowest = min(windows, key=lambda row: row["response_ratio"]) if windows else None
            highest = max(windows, key=lambda row: row["response_ratio"]) if windows else None
            episodes.append({
                "episode_id": f"{suite}-task{task_id:02d}-episode{episode_idx:03d}",
                "suite": suite,
                "task_id": task_id,
                "episode_idx": episode_idx,
                "seed": starts[episode_idx].get("seed"),
                "sampling_noise_seed": starts[episode_idx].get("sampling_noise_seed"),
                "success": success,
                "steps": len(steps),
                "trace_path": str(trace.resolve()),
                "sidecar_status": "matched" if len(matches) == 1 else "missing" if not matches else "ambiguous",
                "sidecar_path": matches[0]["path"] if len(matches) == 1 else None,
                "split": args.force_split or grouped_split(suite, task_id, args.split_seed),
                "label_blind_low_response_window": lowest,
                "label_blind_high_response_window": highest,
            })

    result = {
        "schema_version": 1,
        "split_seed": args.split_seed,
        "split_unit": "entire (suite, task) group",
        "window_selection_firewall": [
            "window selection uses intended/actual translation only",
            "success is retained for later evaluation but is not used to select windows",
            "sealed transfer data is absent from this manifest",
        ],
        "counts": {
            "episodes": len(episodes),
            "success": dict(Counter(str(row["success"]) for row in episodes)),
            "sidecar_status": dict(Counter(row["sidecar_status"] for row in episodes)),
            "split": dict(Counter(row["split"] for row in episodes)),
            "task_groups": len({(row["suite"], row["task_id"]) for row in episodes}),
        },
        "episodes": episodes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(result["counts"], ensure_ascii=False))
    for group in sorted({(row["suite"], row["task_id"], row["split"]) for row in episodes}):
        print("GROUP", *group)


if __name__ == "__main__":
    main()
