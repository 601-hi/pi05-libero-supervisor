import collections
import json
import pathlib
import sys

import numpy as np


path = pathlib.Path(
    sys.argv[1]
    if len(sys.argv) > 1
    else "outputs/joint_history_nullspace_gpu_v23_20261002/mvp.jsonl"
)
rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
steps = [row for row in rows if row.get("event") == "step"]

print("EVENT_COUNTS", dict(collections.Counter(row.get("event") for row in rows)))
print("STEP_COUNT", len(steps))
for key in ("rollback_state", "rollback_phase"):
    print(key.upper(), dict(collections.Counter(str(row.get(key)) for row in steps)))

for key in (
    "actual_spatial_retreat_m",
    "actual_target_joint_error_rad",
    "target_joint_error_rad",
    "rollback_target_action_index",
    "temporary_replay_escape_total_steps",
):
    values = np.asarray(
        [row[key] for row in rows if isinstance(row.get(key), (int, float))], dtype=float
    )
    if values.size:
        print(
            key.upper(),
            {
                "count": int(values.size),
                "min": float(values.min()),
                "median": float(np.median(values)),
                "max": float(values.max()),
                "last": float(values[-1]),
            },
        )

responses = [row for row in rows if row.get("event") == "complex_rollback_response"]
if responses:
    derived_joint_errors = []
    for row in responses:
        target = row.get("rollback_target_joint")
        actual = row.get("joint_pos_after") or row.get("joint_pos")
        if target is not None and actual is not None:
            derived_joint_errors.append(float(np.linalg.norm(np.asarray(actual) - np.asarray(target))))
    if derived_joint_errors:
        derived = np.asarray(derived_joint_errors)
        print(
            "DERIVED_TARGET_JOINT_ERROR_RAD",
            {
                "count": int(derived.size),
                "min": float(derived.min()),
                "median": float(np.median(derived)),
                "max": float(derived.max()),
                "last": float(derived[-1]),
            },
        )
    keys = (
        "action_index",
        "rollback_state",
        "rollback_phase",
        "rollback_target_action_index",
        "actual_spatial_retreat_m",
        "actual_target_joint_error_rad",
        "target_joint_error_rad",
        "swept_collision_free",
        "actual_self_clearance_m",
        "terminal_reason",
    )
    print("LAST_RESPONSES")
    for row in responses[-10:]:
        print({key: row.get(key) for key in keys if key in row})

ends = [row for row in rows if row.get("event") == "episode_end"]
print("EPISODE_END", ends)
