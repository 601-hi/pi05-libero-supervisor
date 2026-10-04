#!/usr/bin/env python3
"""Generate an order-locked blank target-mask annotation document."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_template(prediction_document: dict) -> dict:
    records = prediction_document["records"]
    return {
        "schema_version": 1,
        "protocol": (
            "Annotate only from initial RGB mask overlays. Do not inspect trajectories, "
            "locked_candidate_id, success, reward, or disturbance metadata. List every "
            "candidate mask that adequately covers the manipulated target."
        ),
        "records": [
            {
                "anonymous_id": row["anonymous_id"],
                "acceptable_target_mask_ids": [],
                "target_mask_observable": None,
                "confidence": None,
                "notes": "",
            }
            for row in records
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    document = json.loads(args.predictions.read_text(encoding="utf-8"))
    result = build_template(document)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "records": len(result["records"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
