#!/usr/bin/env python3
"""Evaluate frozen causal identity predictions without tuning thresholds."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


VALID_OBSERVABILITY = {"yes", "no"}


def evaluate(public: dict, annotations: dict, predictions: dict) -> dict:
    expected_ids = {row["anonymous_id"] for row in public["records"]}
    labels = {row["anonymous_id"]: row for row in annotations["records"]}
    preds = {row["anonymous_id"]: row for row in predictions["records"]}
    if set(labels) != expected_ids or set(preds) != expected_ids:
        raise ValueError("annotation and prediction ids must exactly match the frozen manifest")

    rows = []
    for manifest_row in public["records"]:
        episode_id = manifest_row["anonymous_id"]
        label = labels[episode_id]
        pred = preds[episode_id]
        if "target_mask_observable" in label:
            observable_bool = bool(label["target_mask_observable"])
            acceptable_ids = {int(value) for value in label.get("acceptable_manipulated_candidate_ids", [])}
            if observable_bool and not acceptable_ids:
                raise ValueError(f"observable annotation has no acceptable candidate for {episode_id}")
        else:
            observable = label["identity_observable"]
            if observable not in VALID_OBSERVABILITY:
                raise ValueError(f"unresolved annotation for {episode_id}")
            observable_bool = observable == "yes"
            acceptable_ids = {int(label["manipulated_candidate_id"])} if observable_bool else set()
        predicted_id = pred.get("locked_candidate_id")
        correct = observable_bool and predicted_id in acceptable_ids
        false_lock = predicted_id is not None and not correct
        delay = None
        if correct and label.get("first_observable_frame") is not None:
            delay = max(0, int(pred["lock_frame"]) - int(label["first_observable_frame"]))
        rows.append(
            {
                "anonymous_id": episode_id,
                "family": manifest_row["family"],
                "wave": manifest_row["evaluation_wave"],
                "observable": observable_bool,
                "acceptable_candidate_ids": sorted(acceptable_ids),
                "locked": predicted_id is not None,
                "correct_lock": correct,
                "false_lock": false_lock,
                "delay_frames": delay,
            }
        )

    observable_rows = [row for row in rows if row["observable"]]
    delays = [row["delay_frames"] for row in rows if row["delay_frames"] is not None]
    return {
        "schema_version": 1,
        "episodes": len(rows),
        "observable_episodes": len(observable_rows),
        "coverage_on_observable": sum(row["locked"] for row in observable_rows) / max(len(observable_rows), 1),
        "correct_lock_rate_on_observable": sum(row["correct_lock"] for row in observable_rows) / max(len(observable_rows), 1),
        "false_lock_rate_all": sum(row["false_lock"] for row in rows) / max(len(rows), 1),
        "unknown_rate_all": sum(not row["locked"] for row in rows) / max(len(rows), 1),
        "mean_delay_frames_when_correct": sum(delays) / len(delays) if delays else None,
        "records": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(
        json.loads(args.manifest.read_text(encoding="utf-8")),
        json.loads(args.annotations.read_text(encoding="utf-8")),
        json.loads(args.predictions.read_text(encoding="utf-8")),
    )
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
