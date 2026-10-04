#!/usr/bin/env python3
"""Build outcome-blind fixed-view motion groups from relative close tracks."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def point(value):
    if value is None:
        return None
    result = np.asarray(value, dtype=float)
    return result if result.shape == (2,) and np.all(np.isfinite(result)) else None


def summarize(pre_values: list, post_values: list) -> dict:
    pre = [point(value) for value in pre_values]
    post = [point(value) for value in post_values]
    pre_steps = [
        float(np.linalg.norm(second - first))
        for first, second in zip(pre[:-1], pre[1:])
        if first is not None and second is not None
    ]
    noise = float(np.median(pre_steps)) if pre_steps else 0.0
    threshold = max(3.0, 4.0 * noise)
    origin = post[0] if post else None
    distances = [
        (index, float(np.linalg.norm(value - origin)))
        for index, value in enumerate(post)
        if origin is not None and value is not None
    ]
    tail = [value for _, value in distances[-min(5, len(distances)):]]
    return {
        "preclose_median_step_px": noise,
        "adaptive_motion_threshold_px": threshold,
        "postclose_visible_fraction": sum(value is not None for value in post) / max(len(post), 1),
        "postclose_net_displacement_px": distances[-1][1] if distances else None,
        "postclose_max_displacement_px": max((value for _, value in distances), default=None),
        "motion_onset_relative_frame": next((index for index, value in distances if value >= threshold), None),
        "sustained_motion": bool(tail and np.median(tail) >= threshold),
    }


def groups(sequences: dict[str, list], identifiers: set[str], threshold: float = 4.0) -> list[list[str]]:
    identifiers = sorted(identifiers)
    parent = {value: value for value in identifiers}

    def find(value):
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    for index, first_id in enumerate(identifiers):
        first = [point(value) for value in sequences[first_id]]
        for second_id in identifiers[index + 1:]:
            second = [point(value) for value in sequences[second_id]]
            differences = []
            if not first or not second or first[0] is None or second[0] is None:
                continue
            for first_value, second_value in zip(first, second):
                if first_value is not None and second_value is not None:
                    differences.append(float(np.linalg.norm(
                        (first_value - first[0]) - (second_value - second[0])
                    )))
            if len(differences) >= 5 and np.median(differences) <= threshold:
                parent[find(second_id)] = find(first_id)
    result: dict[str, list[str]] = {}
    for identifier in identifiers:
        result.setdefault(find(identifier), []).append(identifier)
    return sorted(result.values(), key=lambda value: (-len(value), value))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    document = json.loads(args.predictions.read_text(encoding="utf-8"))
    if document.get("protocol", {}).get("camera_key") != "agent_images":
        raise ValueError("fixed-view audit requires camera_key=agent_images")
    rows = []
    for record in document["records"]:
        if record.get("close_frame") is None:
            rows.append({"anonymous_id": record["anonymous_id"], "events": []})
            continue
        candidates = []
        for candidate_id in record["post_centroids_xy"]:
            evidence = summarize(
                record["pre_centroids_xy"][candidate_id],
                record["post_centroids_xy"][candidate_id],
            )
            candidates.append({"candidate_id": candidate_id, **evidence})
        candidates.sort(key=lambda value: (
            value["sustained_motion"], value["postclose_max_displacement_px"] or -1
        ), reverse=True)
        sustained = {row["candidate_id"] for row in candidates if row["sustained_motion"]}
        rows.append({
            "anonymous_id": record["anonymous_id"],
            "events": [{
                "close_frame": record["close_frame"],
                "co_motion_groups": groups(record["post_centroids_xy"], sustained),
                "candidates": candidates,
            }],
        })
    result = {
        "schema_version": 1,
        "scope": "outcome-blind fixed-view motion evidence; not object identity",
        "forbidden_inputs": ["task language", "episode outcome", "reward"],
        "records": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "episodes": len(rows),
        "with_motion_groups": sum(bool(row["events"] and row["events"][0]["co_motion_groups"]) for row in rows),
    }, indent=2))


if __name__ == "__main__":
    main()
