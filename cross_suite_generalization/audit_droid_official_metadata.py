#!/usr/bin/env python3
"""Fetch and validate only the two tiny official DROID-100 TFDS metadata files."""
from __future__ import annotations

import base64
import hashlib
import json
import urllib.request


BASE = "https://storage.googleapis.com/gresearch/robotics/droid_100/1.0.0/"
EXPECTED = {
    "dataset_info.json": (760, "JOcTGdwePw+iymeLhrqavg=="),
    "features.json": (18665, "AUayWu2IHU/GMy61byaQpg=="),
}


def fetch(name: str) -> dict:
    payload = urllib.request.urlopen(BASE + name, timeout=20).read()
    expected_size, expected_md5 = EXPECTED[name]
    actual_md5 = base64.b64encode(hashlib.md5(payload).digest()).decode("ascii")
    if len(payload) != expected_size or actual_md5 != expected_md5:
        raise ValueError(f"official metadata identity changed for {name}")
    return json.loads(payload)


def main() -> None:
    info = fetch("dataset_info.json")
    features = fetch("features.json")
    root = features["featuresDict"]["features"]
    step = root["steps"]["sequence"]["feature"]["featuresDict"]["features"]
    action = step["action_dict"]["featuresDict"]["features"]
    observation = step["observation"]["featuresDict"]["features"]
    required_actions = {
        "cartesian_position": 6,
        "cartesian_velocity": 6,
        "joint_position": 7,
        "joint_velocity": 7,
        "gripper_position": 1,
        "gripper_velocity": 1,
    }
    required_observations = {"cartesian_position": 6, "joint_position": 7, "gripper_position": 1}
    for fields, required in ((action, required_actions), (observation, required_observations)):
        for name, dimension in required.items():
            actual = int(fields[name]["tensor"]["shape"]["dimensions"][0])
            if actual != dimension:
                raise ValueError(f"unexpected dimension for {name}: {actual}")
    absent = sorted({"joint_velocity", "motor_torques_measured", "timestamp"} - set(observation))
    report = {
        "ok": True,
        "dataset_name": info["name"],
        "version": info["version"],
        "episodes": sum(int(value) for split in info["splits"] for value in split["shardLengths"]),
        "action_dict_dimensions": required_actions,
        "observation_dimensions": required_observations,
        "measured_fields_absent_from_rlds": absent,
        "flat_action_description": step["action"].get("description", ""),
        "flat_action_forbidden": True,
        "training_use_allowed_by_metadata_audit_alone": False,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
