"""Paired outcome accounting for recovery interventions.

Pairs must share task, initial state, policy-sampling noise and disturbance.
The treatment arm differs only by whether recovery is allowed to take control.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np


@dataclass(frozen=True)
class PairedRecoverySummary:
    pairs: int
    control_successes: int
    treatment_successes: int
    recovered_failures: int
    intervention_harms: int
    both_succeeded: int
    both_failed: int
    success_rate_delta: float
    delta_ci95: tuple[float, float]
    mcnemar_exact_p: float
    treatment_safe_stops: int
    treatment_hard_faults: int
    mean_action_overhead: float


def _exact_two_sided_binomial_p(left: int, right: int) -> float:
    discordant = int(left + right)
    if discordant == 0:
        return 1.0
    extreme = min(int(left), int(right))
    tail = sum(math.comb(discordant, k) for k in range(extreme + 1)) / (2 ** discordant)
    return float(min(1.0, 2.0 * tail))


def summarize_paired_recovery(rows: Iterable[Mapping], *,
                              bootstrap_samples: int = 5000,
                              bootstrap_seed: int = 20260925
                              ) -> PairedRecoverySummary:
    records = list(rows)
    if not records:
        raise ValueError("paired evaluation requires at least one pair")
    required = {
        "pair_id", "task_id", "initial_state_id", "policy_noise_id",
        "disturbance_id", "control_success", "treatment_success",
        "control_actions", "treatment_actions", "treatment_safe_stop",
        "treatment_hard_fault",
    }
    pair_ids = []
    deltas = []
    for row in records:
        missing = required.difference(row)
        if missing:
            raise ValueError(f"paired row missing fields: {sorted(missing)}")
        pair_ids.append(str(row["pair_id"]))
        deltas.append(int(bool(row["treatment_success"]))
                      - int(bool(row["control_success"])))
    if len(pair_ids) != len(set(pair_ids)):
        raise ValueError("pair_id values must be unique")

    control = np.asarray([bool(row["control_success"]) for row in records])
    treatment = np.asarray([bool(row["treatment_success"]) for row in records])
    delta = np.asarray(deltas, dtype=float)
    recovered = int(np.sum(~control & treatment))
    harmed = int(np.sum(control & ~treatment))
    rng = np.random.default_rng(bootstrap_seed)
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    indexes = rng.integers(0, len(records), size=(bootstrap_samples, len(records)))
    bootstrap = np.mean(delta[indexes], axis=1)
    overhead = np.asarray([
        int(row["treatment_actions"]) - int(row["control_actions"])
        for row in records
    ], dtype=float)
    return PairedRecoverySummary(
        pairs=len(records),
        control_successes=int(np.sum(control)),
        treatment_successes=int(np.sum(treatment)),
        recovered_failures=recovered,
        intervention_harms=harmed,
        both_succeeded=int(np.sum(control & treatment)),
        both_failed=int(np.sum(~control & ~treatment)),
        success_rate_delta=float(np.mean(delta)),
        delta_ci95=(float(np.percentile(bootstrap, 2.5)),
                    float(np.percentile(bootstrap, 97.5))),
        mcnemar_exact_p=_exact_two_sided_binomial_p(recovered, harmed),
        treatment_safe_stops=sum(bool(row["treatment_safe_stop"]) for row in records),
        treatment_hard_faults=sum(bool(row["treatment_hard_fault"]) for row in records),
        mean_action_overhead=float(np.mean(overhead)),
    )


def _episode_end(path: Path) -> Mapping:
    endings = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                if row.get("event") == "episode_end":
                    endings.append(row)
    if len(endings) != 1:
        raise ValueError(f"expected exactly one episode_end in {path}, got {len(endings)}")
    return endings[0]


def evaluate_manifest(manifest: Mapping, root: Path) -> dict:
    """Load the paired trace outcomes referenced by a frozen-run manifest."""
    rows = []
    details = []
    for pair in manifest.get("pairs", []):
        baseline = _episode_end(root / str(pair["baseline_trace"]))
        treatment = _episode_end(root / str(pair["control_trace"]))
        counts = treatment.get("supervisor_intervention_counts", {})
        row = {
            "pair_id": str(pair["pair_id"]),
            "task_id": f'{pair.get("suite", "unknown")}:{pair["task_id"]}',
            "initial_state_id": f'seed:{pair["seed"]}',
            "policy_noise_id": f'sampling_noise_seed:{pair["sampling_noise_seed"]}',
            "disturbance_id": str(pair.get("disturbance_id", "none")),
            "control_success": bool(baseline["success"]),
            "treatment_success": bool(treatment["success"]),
            "control_actions": int(baseline["executed_actions"]),
            "treatment_actions": int(treatment["executed_actions"]),
            "treatment_safe_stop": bool(counts.get("safe_stop", 0)),
            "treatment_hard_fault": bool(treatment.get("supervisor_hard_fault", False)),
        }
        rows.append(row)
        details.append({
            **row,
            "treatment_replans": int(treatment.get("supervisor_replans", 0)),
            "rollback_executed_steps": int(treatment.get("rollback_executed_steps", 0)),
            "intervention_counts": counts,
        })
    summary = asdict(summarize_paired_recovery(rows))
    summary["delta_ci95"] = list(summary["delta_ci95"])
    summary["pair_details"] = details
    return summary
