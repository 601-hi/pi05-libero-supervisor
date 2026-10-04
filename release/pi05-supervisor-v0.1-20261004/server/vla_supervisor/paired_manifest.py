"""Outcome-blind construction and validation of paired recovery manifests."""
from __future__ import annotations

import hashlib
from typing import Iterable, Mapping

import numpy as np


DEFAULT_SUITE_SIZES = {
    "libero_spatial": 10,
    "libero_object": 10,
    "libero_goal": 10,
    "libero_10": 10,
    "libero_90": 90,
}


def evenly_spaced_task_ids(task_count: int, sample_count: int) -> tuple[int, ...]:
    if task_count < 1 or not 1 <= sample_count <= task_count:
        raise ValueError("require 1 <= sample_count <= task_count")
    raw = np.linspace(0, task_count - 1, sample_count)
    return tuple(sorted(set(int(round(value)) for value in raw)))


def build_paired_manifest(*, suites: Iterable[str], tasks_per_suite: int,
                          initial_state_ids: Iterable[int],
                          policy_noise_seeds: Iterable[int],
                          disturbance_id: str = "none") -> list[dict]:
    records = []
    for suite in suites:
        if suite not in DEFAULT_SUITE_SIZES:
            raise ValueError(f"unknown suite: {suite}")
        task_ids = evenly_spaced_task_ids(
            DEFAULT_SUITE_SIZES[suite], tasks_per_suite)
        for task_id in task_ids:
            for initial_state_id in initial_state_ids:
                for noise_seed in policy_noise_seeds:
                    identity = (
                        f"{suite}|{task_id}|{int(initial_state_id)}|"
                        f"{int(noise_seed)}|{disturbance_id}")
                    pair_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]
                    base = {
                        "pair_id": pair_id,
                        "suite": suite,
                        "task_id": int(task_id),
                        "initial_state_id": int(initial_state_id),
                        "policy_noise_id": str(int(noise_seed)),
                        "policy_noise_seed": int(noise_seed),
                        "disturbance_id": str(disturbance_id),
                        "selection_policy": "deterministic_even_spacing_without_outcome_filtering",
                    }
                    records.extend((
                        {**base, "arm": "control", "recovery_enabled": False},
                        {**base, "arm": "treatment", "recovery_enabled": True},
                    ))
    validate_paired_manifest(records)
    return records


def validate_paired_manifest(rows: Iterable[Mapping]) -> None:
    records = list(rows)
    if not records:
        raise ValueError("manifest must not be empty")
    by_pair: dict[str, list[Mapping]] = {}
    for row in records:
        by_pair.setdefault(str(row.get("pair_id")), []).append(row)
    identity_fields = (
        "suite", "task_id", "initial_state_id", "policy_noise_id",
        "disturbance_id", "selection_policy")
    for pair_id, pair in by_pair.items():
        if len(pair) != 2 or {row.get("arm") for row in pair} != {"control", "treatment"}:
            raise ValueError(f"pair {pair_id} must contain exactly control and treatment")
        for field in identity_fields:
            if pair[0].get(field) != pair[1].get(field):
                raise ValueError(f"pair {pair_id} mismatches {field}")
        control = next(row for row in pair if row["arm"] == "control")
        treatment = next(row for row in pair if row["arm"] == "treatment")
        if bool(control.get("recovery_enabled")) or not bool(treatment.get("recovery_enabled")):
            raise ValueError(f"pair {pair_id} has invalid recovery assignment")
