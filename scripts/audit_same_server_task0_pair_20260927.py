import json
import pathlib

import numpy as np


ROOT = pathlib.Path(
    "/root/gpufree-data/libero-traces/natural_failure_same_server_pair_20260927"
)


def read(root, name):
    return [json.loads(line) for line in (root / name).open(encoding="utf-8") if line.strip()]


def audit_rows(baseline, control):
    baseline_steps = [row for row in baseline if row.get("event") == "step"]
    control_steps = [row for row in control if row.get("event") == "step"]
    rollback = [row for row in control if row.get("event") == "complex_rollback_proposal"]
    first_intervention = rollback[0]["action_index"] if rollback else None
    if first_intervention is None:
        raise SystemExit("control never entered complex rollback")

    # The proposal for action k is allowed to change action k.  Therefore the
    # strict causal prefix is [0, k), not [0, k].
    prefix = int(first_intervention)
    if len(baseline_steps) < prefix or len(control_steps) < prefix:
        raise ValueError("one trace is shorter than the claimed causal prefix")
    expected_indices = list(range(prefix))
    if [row.get("action_index") for row in baseline_steps[:prefix]] != expected_indices:
        raise ValueError("baseline action indices are not a complete causal prefix")
    if [row.get("action_index") for row in control_steps[:prefix]] != expected_indices:
        raise ValueError("control action indices are not a complete causal prefix")
    fields = (
        "intended_action", "executed_action", "eef_pos_before", "eef_pos_after",
        "joint_pos_before", "joint_pos_after",
    )
    report = {}
    for field in fields:
        maximum = 0.0
        first = None
        for index, (left, right) in enumerate(
                zip(baseline_steps[:prefix], control_steps[:prefix])):
            difference = float(np.max(np.abs(
                np.asarray(left[field], dtype=float)
                - np.asarray(right[field], dtype=float))))
            maximum = max(maximum, difference)
            if first is None and difference != 0.0:
                first = index
        report[field] = {"maximum_absolute_difference": maximum,
                         "first_difference_index": first}

    baseline_inference = [row for row in baseline if row.get("event") == "inference"
                          and int(row.get("t", 0)) < prefix]
    control_inference = [row for row in control if row.get("event") == "inference"
                         and int(row.get("t", 0)) < prefix]
    noise_equal = (
        len(baseline_inference) > 0
        and len(baseline_inference) == len(control_inference)
        and
        [row.get("sampling_noise_sha256") for row in baseline_inference]
        == [row.get("sampling_noise_sha256") for row in control_inference]
    )
    ends = {
        "baseline": [row for row in baseline if row.get("event") == "episode_end"],
        "control": [row for row in control if row.get("event") == "episode_end"],
    }
    exact = all(item["maximum_absolute_difference"] == 0.0 for item in report.values())
    if len(ends["baseline"]) != 1 or len(ends["control"]) != 1:
        raise ValueError("each trace must contain exactly one episode_end")
    return {
        "first_intervention_action": first_intervention,
        "strict_prefix_steps": prefix,
        "noise_hash_prefix_equal": noise_equal,
        "strict_prefix_exact": exact,
        "fields": report,
        "episode_end": ends,
    }


def audit_directory(root):
    baseline = read(root, "libero90_t0_seed34_baseline.jsonl")
    control = read(root, "libero90_t0_seed34_control.jsonl")
    result = audit_rows(baseline, control)
    (root / "paired_causal_audit.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


def main():
    result = audit_directory(ROOT)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if not result["noise_hash_prefix_equal"] or not result["strict_prefix_exact"]:
        raise SystemExit("paired causal prefix is not exact; effectiveness comparison invalid")


if __name__ == "__main__":
    main()
