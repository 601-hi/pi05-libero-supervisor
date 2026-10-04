"""Emit a preregistered, disk-aware expanded evaluation design."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total == 0:
        return 0.0, 1.0
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return centre - radius, centre + radius


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tasks = {
        "libero_spatial": [0, 2, 4, 6, 8],
        "libero_object": [0, 2, 4, 6, 8],
        "libero_goal": [0, 2, 4, 6, 8],
        "libero_90": [0, 9, 19, 29, 39],
    }
    tiers = [
        {"name": "development", "seed": 36, "noise_seed": 2026091836},
        {"name": "model_selection", "seed": 37, "noise_seed": 2026091837},
        {"name": "sealed_final_test", "seed": 38, "noise_seed": 2026091838},
    ]
    trials = 4
    episodes_per_tier = sum(len(value) for value in tasks.values()) * trials
    expected_mb_per_episode = 18.5
    result = {
        "schema_version": 1,
        "preregistered_before_collection": True,
        "primary_unit": "episode; report cluster breakdown by suite/task/seed",
        "tasks": tasks,
        "trials_per_task": trials,
        "tiers": tiers,
        "episodes_per_tier": episodes_per_tier,
        "total_planned_episodes": episodes_per_tier * len(tiers),
        "estimated_visual_storage_gib": round(
            episodes_per_tier * len(tiers) * expected_mb_per_episode / 1024, 2
        ),
        "storage_policy": {
            "minimum_free_gib_before_each_tier": 8,
            "never_delete_automatically": True,
            "stop_if_estimated_post_run_free_gib_below": 5,
        },
        "firewall": {
            "development": "fit features/models and inspect errors",
            "model_selection": "choose one frozen model and temporal rule",
            "sealed_final_test": "no fitting, threshold choice, task choice, or case inspection before scoring",
        },
        "precision_examples_95pct_wilson": {
            "current_6_of_6_failure_detection": wilson(6, 6),
            "hypothetical_45_of_50_failure_detection": wilson(45, 50),
            "hypothetical_90_of_100_normal_specificity": wilson(90, 100),
        },
        "stopping_rule": (
            "Do not claim stable performance until at least 50 independent natural-failure episodes "
            "and 100 independent normal episodes have been evaluated outside fitting data; otherwise "
            "report Wilson intervals and call the result preliminary."
        ),
        "preregistered_extension": {
            "trigger": (
                "After scoring seed38, if model-selection plus final-test data contain fewer than 50 "
                "natural-failure episodes or fewer than 100 normal episodes, collect seed39 using the "
                "identical task matrix. The trigger may inspect outcome counts only, never model scores."
            ),
            "name": "sealed_confirmation",
            "seed": 39,
            "noise_seed": 2026091839,
            "maximum_extra_episodes": episodes_per_tier,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
