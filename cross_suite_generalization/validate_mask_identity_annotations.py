#!/usr/bin/env python3
"""Validate human target-mask equivalence labels before evaluation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def validate(prediction_document: dict, annotation_document: dict) -> list[str]:
    predictions = prediction_document.get("records", [])
    annotations = annotation_document.get("records", [])
    errors: list[str] = []
    if not isinstance(predictions, list) or not isinstance(annotations, list):
        return ["predictions.records and annotations.records must be lists"]
    if any(not isinstance(row, dict) or "anonymous_id" not in row for row in predictions + annotations):
        return ["every record must be an object containing anonymous_id"]
    predicted_ids = [row["anonymous_id"] for row in predictions]
    annotated_ids = [row["anonymous_id"] for row in annotations]
    if len(predicted_ids) != len(set(predicted_ids)):
        errors.append("predictions contain duplicate anonymous IDs")
    if len(annotated_ids) != len(set(annotated_ids)):
        errors.append("annotations contain duplicate anonymous IDs")
    if predicted_ids != annotated_ids:
        errors.append("annotation IDs/order do not exactly match predictions")
    by_id = {row["anonymous_id"]: row for row in predictions}
    allowed_confidence = {"low", "medium", "high"}
    for row in annotations:
        identifier = row["anonymous_id"]
        if identifier not in by_id:
            continue
        candidate_count = int(by_id[identifier]["num_candidates"])
        target_ids = row.get("acceptable_target_mask_ids")
        if not isinstance(target_ids, list):
            errors.append(f"{identifier}: acceptable_target_mask_ids must be a list")
            continue
        if len(target_ids) != len(set(target_ids)):
            errors.append(f"{identifier}: duplicate target mask IDs")
        invalid = [value for value in target_ids if not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= candidate_count]
        if invalid:
            errors.append(f"{identifier}: invalid IDs {invalid}; candidate range is 1..{candidate_count}")
        observable = row.get("target_mask_observable")
        if not isinstance(observable, bool):
            errors.append(f"{identifier}: target_mask_observable must be boolean")
        elif observable and not target_ids:
            errors.append(f"{identifier}: observable target requires at least one mask ID")
        elif not observable and target_ids:
            errors.append(f"{identifier}: unobservable target must have an empty mask ID list")
        if row.get("confidence") not in allowed_confidence:
            errors.append(f"{identifier}: confidence must be low, medium, or high")
        if not isinstance(row.get("notes", ""), str):
            errors.append(f"{identifier}: notes must be a string")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    args = parser.parse_args()
    prediction_document = json.loads(args.predictions.read_text(encoding="utf-8"))
    annotation_document = json.loads(args.annotations.read_text(encoding="utf-8"))
    errors = validate(prediction_document, annotation_document)
    result = {"episodes": len(prediction_document.get("records", [])), "annotations": len(annotation_document.get("records", [])), "errors": errors, "valid": not errors}
    print(json.dumps(result, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
