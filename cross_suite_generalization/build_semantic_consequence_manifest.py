"""Build label-blind semantic checkpoints with a strict runtime/truth firewall."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import numpy as np
from vla_supervisor.goal_relations import parse_goal_relation


GOAL_RE = re.compile(r"\((On|In|Open|Close|Turnon|Turnoff)\s+([^()]+)\)", re.IGNORECASE)


def runtime_family(language: str) -> str:
    relation = parse_goal_relation(language).relation
    return {'turn_on': 'binary_state_change', 'turn_off': 'binary_state_change',
            'open': 'articulated_state_change', 'close': 'articulated_state_change',
            'place_in': 'containment_relation', 'place_on': 'support_relation'}.get(
                relation, 'unknown_goal_relation')


def parse_offline_goal(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    goal = text[text.lower().find("(:goal"):]
    return [
        {"predicate": match.group(1), "arguments": match.group(2).split()}
        for match in GOAL_RE.finditer(goal)
    ]


def trace_steps(path: Path, episode_idx: int) -> list[dict]:
    return [
        row for line in path.open(encoding="utf-8")
        if (row := json.loads(line)).get("event") == "step"
        and int(row["episode_idx"]) == episode_idx
    ]


def gripper_transitions(steps: list[dict], minimum_gap: int = 10, maximum: int = 4) -> list[int]:
    signs = np.sign([float(row["executed_action"][-1]) for row in steps])
    candidates = [index for index in range(1, len(signs)) if signs[index] != signs[index - 1]]
    chosen = []
    for index in candidates:
        action_index = int(steps[index]["action_index"])
        if not chosen or action_index - chosen[-1] >= minimum_gap:
            chosen.append(action_index)
        if len(chosen) == maximum:
            break
    return chosen


def main() -> None:
    from libero.libero import benchmark
    parser = argparse.ArgumentParser()
    parser.add_argument("--multi-event-manifest", type=Path, required=True)
    parser.add_argument("--libero-bddl-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.multi_event_manifest.read_text(encoding="utf-8"))
    suites = {}
    episodes = []
    for episode in source["episodes"]:
        suite_name = episode["suite"]
        if suite_name not in suites:
            suites[suite_name] = benchmark.get_benchmark_dict()[suite_name]()
        suite = suites[suite_name]
        task = suite.get_task(int(episode["task_id"]))
        steps = trace_steps(Path(episode["trace_path"]), int(episode["episode_idx"]))
        action_indices = [int(row["action_index"]) for row in steps]
        checkpoints = [("start", action_indices[0]), ("end", action_indices[-1])]
        checkpoints.extend((f"gripper_transition_{i}", value)
                           for i, value in enumerate(gripper_transitions(steps)))
        checkpoints.extend((f"motion_event_{i}", int(window["start_action_index"]))
                           for i, window in enumerate(episode["label_blind_low_response_events"]))
        # Preserve semantic names if two reasons point to the same frame.
        checkpoints = [
            {"reason": reason, "action_index": action_index}
            for reason, action_index in checkpoints
        ]
        bddl_path = args.libero_bddl_root / task.problem_folder / task.bddl_file
        episodes.append({
            "episode_id": episode["episode_id"],
            "suite": suite_name, "task_id": int(episode["task_id"]),
            "episode_idx": int(episode["episode_idx"]),
            "sidecar_path": episode["sidecar_path"],
            "trace_path": episode["trace_path"],
            "runtime_inputs": {
                "task_language": task.language,
                "goal_family_from_language": runtime_family(task.language),
                "checkpoints": checkpoints,
            },
            "offline_evaluation_only": {
                "episode_success": bool(episode["success"]),
                "bddl_file": str(bddl_path),
                "goal_predicates": parse_offline_goal(bddl_path),
            },
        })
    payload = {
        "schema_version": 1,
        "information_firewall": {
            "runtime_allowed": ["RGB", "task language", "action history", "gripper state"],
            "evaluation_only": ["episode success", "BDDL predicate", "simulator object state"],
            "prohibition": "evaluation-only fields must never enter a runtime model or threshold",
        },
        "checkpoint_policy": (
            "start/end, up to four gripper sign transitions separated by ten actions, and the two "
            "pre-registered low-response motion events; no outcome-dependent frame selection"
        ),
        "episodes": episodes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "episodes": len(episodes),
        "checkpoints": sum(len(row["runtime_inputs"]["checkpoints"]) for row in episodes),
        "goal_families": sorted({row["runtime_inputs"]["goal_family_from_language"] for row in episodes}),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
