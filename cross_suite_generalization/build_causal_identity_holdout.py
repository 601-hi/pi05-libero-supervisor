#!/usr/bin/env python3
"""Freeze a label-blind holdout for causal manipulated-object identity."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


DEFAULT_SALT = "causal-object-identity-holdout-v1-20260913"


def episode_key(job: dict) -> str:
    return "|".join(
        (
            str(job["suite"]),
            str(job["goal_language"]),
            str(job["task_id_for_data_join_only"]),
            str(job["episode_idx"]),
        )
    )


def digest(key: str, salt: str) -> str:
    return hashlib.sha256(f"{salt}|{key}".encode("utf-8")).hexdigest()


def build(manifest: dict, pilot: dict, salt: str = DEFAULT_SALT, wave_size: int = 20):
    pilot_goals = {row["goal_language"] for row in pilot["episodes"]}
    first_for_goal: dict[str, str] = {}
    for job in manifest["jobs"]:
        first_for_goal.setdefault(job["goal_language"], episode_key(job))
    pilot_keys = {first_for_goal[goal] for goal in pilot_goals}

    records = []
    private = []
    for job in manifest["jobs"]:
        key = episode_key(job)
        if key in pilot_keys:
            continue
        family = "articulated_environment" if job["goal_language"].startswith("close ") else "rigid_object_transport"
        token = digest(key, salt)
        anonymous_id = f"hoi-{token[:12]}"
        private.append(
            {
                "anonymous_id": anonymous_id,
                "episode_key": key,
                "original_sidecar": job["sidecar"],
                "suite": job["suite"],
                "task_id": job["task_id_for_data_join_only"],
                "episode_idx": job["episode_idx"],
                "outcome_for_posthoc_audit_only": job["outcome_for_audit_only"],
            }
        )
        records.append(
            {
                "anonymous_id": anonymous_id,
                "family": family,
                "goal_language": job["goal_language"],
                "selection_digest": token,
            }
        )

    records.sort(key=lambda row: row["selection_digest"])
    rigid_index = 0
    articulated_index = 0
    for row in records:
        if row["family"] == "rigid_object_transport":
            row["evaluation_wave"] = 1 + rigid_index // wave_size
            rigid_index += 1
        else:
            row["evaluation_wave"] = 1 + articulated_index // wave_size
            articulated_index += 1
    public = {
        "schema_version": 1,
        "protocol": "causal manipulated-object identity holdout",
        "selection": "SHA256 ordering fixed before predictions; independent of outcome",
        "data_access": "physical sidecar paths are isolated in the private audit map",
        "salt": salt,
        "pilot_episode_count_excluded": len(pilot_keys),
        "records": records,
        "feature_firewall": {
            "allowed": ["RGB sequence", "goal language", "past/current time index"],
            "forbidden": ["suite", "task id", "success", "reward", "scale", "simulator state"],
        },
    }
    return public, {"schema_version": 1, "records": private}


def annotation_template(public: dict) -> dict:
    return {
        "schema_version": 1,
        "instructions": "Review overlays without outcome metadata. Candidate id or unknown; do not infer success.",
        "records": [
            {
                "anonymous_id": row["anonymous_id"],
                "manipulated_candidate_id": "unlabeled",
                "identity_observable": "unlabeled",
                "first_observable_frame": None,
                "annotator_confidence": "unlabeled",
                "notes": "",
            }
            for row in public["records"]
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--public-output", type=Path, required=True)
    parser.add_argument("--private-output", type=Path, required=True)
    parser.add_argument("--annotation-output", type=Path, required=True)
    parser.add_argument("--wave-size", type=int, default=20)
    args = parser.parse_args()
    if args.wave_size < 1:
        raise ValueError("wave-size must be positive")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    pilot = json.loads(args.pilot.read_text(encoding="utf-8"))
    public, private = build(manifest, pilot, wave_size=args.wave_size)
    annotation = annotation_template(public)
    for path, value in (
        (args.public_output, public),
        (args.private_output, private),
        (args.annotation_output, annotation),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    counts = {}
    for row in public["records"]:
        counts[row["family"]] = counts.get(row["family"], 0) + 1
    print(json.dumps({"episodes": len(public["records"]), "by_family": counts}, indent=2))


if __name__ == "__main__":
    main()
