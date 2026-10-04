"""Outcome-first evaluation for paired baseline/control LIBERO rollouts.

Pairs must share suite, task, episode seed, initial state, and policy-noise seed.
The control trace may diverge only after its first *executed* intervention.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class PairResult:
    pair_id: str
    baseline_success: bool
    control_success: bool
    outcome: str
    first_intervention_index: int | None
    pre_intervention_actions_equal: bool
    pre_intervention_states_equal: bool
    baseline_steps: int
    control_steps: int
    baseline_inferences: int
    control_inferences: int
    control_replans: int
    intervention_counts: dict[str, int]

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def load_trace(path: str | Path) -> tuple[list[dict], dict]:
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]
    steps = [row for row in rows if row.get("event") == "step"]
    ends = [row for row in rows if row.get("event") == "episode_end"]
    if len(ends) != 1:
        raise ValueError(f"{path}: expected exactly one episode_end, got {len(ends)}")
    return steps, ends[0]


def _is_executed_intervention(step: dict) -> bool:
    if "supervisor_control_epoch_before_action" in step:
        return int(step["supervisor_control_epoch_before_action"]) > 0
    source = step.get("supervisor_action_source")
    intervention = step.get("supervisor_intervention") or {}
    mode = intervention.get("mode")
    # Backward-compatible fallback for traces predating the causal epoch field.
    # A post-step directive alone is not enough: it labels the triggering action,
    # which was already executed.  A recovery bridge is the first changed action.
    return source not in (None, "policy_chunk")


def _first_intervention(steps: Iterable[dict]) -> int | None:
    for row in steps:
        if _is_executed_intervention(row):
            return int(row["action_index"])
    return None


def _first_intervention_position(steps: Iterable[dict]) -> int | None:
    """Return the zero-based row position, independent of trace index origin."""
    for position, row in enumerate(steps):
        if _is_executed_intervention(row):
            return position
    return None


def _outcome(baseline_success: bool, control_success: bool) -> str:
    if not baseline_success and control_success:
        return "rescued"
    if baseline_success and not control_success:
        return "harmed"
    if baseline_success and control_success:
        return "preserved_success"
    return "unresolved_failure"


def evaluate_pair(pair_id: str, baseline_path: str | Path,
                  control_path: str | Path, atol: float = 0.0) -> PairResult:
    baseline_steps, baseline_end = load_trace(baseline_path)
    control_steps, control_end = load_trace(control_path)
    first = _first_intervention(control_steps)
    first_position = _first_intervention_position(control_steps)
    prefix = (min(len(baseline_steps), len(control_steps))
              if first_position is None else first_position)
    baseline_actions = np.asarray(
        [row.get("intended_action", row.get("action")) for row in baseline_steps[:prefix]], float)
    control_actions = np.asarray(
        [row.get("intended_action", row.get("action")) for row in control_steps[:prefix]], float)
    same_actions = baseline_actions.shape == control_actions.shape and np.allclose(
        baseline_actions, control_actions, rtol=0.0, atol=atol)
    if first is None:
        same_actions = same_actions and len(baseline_steps) == len(control_steps)
    baseline_states = np.asarray(
        [row["eef_pos_before"] for row in baseline_steps[:prefix]
         if "eef_pos_before" in row], float)
    control_states = np.asarray(
        [row["eef_pos_before"] for row in control_steps[:prefix]
         if "eef_pos_before" in row], float)
    same_states = baseline_states.shape == control_states.shape and np.allclose(
        baseline_states, control_states, rtol=0.0, atol=atol)
    b_success = bool(baseline_end["success"])
    c_success = bool(control_end["success"])
    return PairResult(
        pair_id=pair_id,
        baseline_success=b_success,
        control_success=c_success,
        outcome=_outcome(b_success, c_success),
        first_intervention_index=first,
        pre_intervention_actions_equal=bool(same_actions),
        pre_intervention_states_equal=bool(same_states),
        baseline_steps=int(baseline_end.get("executed_actions", len(baseline_steps))),
        control_steps=int(control_end.get("executed_actions", len(control_steps))),
        baseline_inferences=int(baseline_end.get("inference_calls", 0)),
        control_inferences=int(control_end.get("inference_calls", 0)),
        control_replans=int(control_end.get("supervisor_replans", 0)),
        intervention_counts={str(k): int(v) for k, v in
                             control_end.get("supervisor_intervention_counts", {}).items()},
    )


def evaluate_manifest(manifest: dict, root: str | Path = ".") -> dict:
    root = Path(root)
    results = [evaluate_pair(
        str(pair["pair_id"]), root / pair["baseline_trace"], root / pair["control_trace"])
        for pair in manifest["pairs"]]
    counts = Counter(result.outcome for result in results)
    causal_identity_failures = sum(
        not (result.pre_intervention_actions_equal and result.pre_intervention_states_equal)
        for result in results
    )
    return {
        "schema_version": 1,
        "pair_count": len(results),
        "outcomes": dict(counts),
        "rescue_rate_among_baseline_failures": (
            counts["rescued"] / (counts["rescued"] + counts["unresolved_failure"])
            if counts["rescued"] + counts["unresolved_failure"] else None
        ),
        "harm_rate_among_baseline_successes": (
            counts["harmed"] / (counts["harmed"] + counts["preserved_success"])
            if counts["harmed"] + counts["preserved_success"] else None
        ),
        "causal_identity_failures": causal_identity_failures,
        "valid_for_causal_claims": causal_identity_failures == 0,
        "step_delta_control_minus_baseline": int(sum(
            result.control_steps - result.baseline_steps for result in results)),
        "inference_delta_control_minus_baseline": int(sum(
            result.control_inferences - result.baseline_inferences for result in results)),
        "pairs": [result.to_dict() for result in results],
    }
