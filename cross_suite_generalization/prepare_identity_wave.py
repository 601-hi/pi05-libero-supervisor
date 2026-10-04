"""Prepare matching public/private manifests for one preregistered wave."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--public", type=Path, required=True)
    parser.add_argument("--private", type=Path, required=True)
    parser.add_argument("--wave", type=int, required=True)
    parser.add_argument("--family", default="rigid_object_transport")
    parser.add_argument("--public-output", type=Path, required=True)
    parser.add_argument("--private-output", type=Path, required=True)
    args = parser.parse_args()

    public = json.loads(args.public.read_text(encoding="utf-8"))
    private = json.loads(args.private.read_text(encoding="utf-8"))
    rows = [
        row for row in public["records"]
        if row["evaluation_wave"] == args.wave and row["family"] == args.family
    ]
    ids = {row["anonymous_id"] for row in rows}
    private_rows = [row for row in private["records"] if row["anonymous_id"] in ids]
    if len(private_rows) != len(rows) or {r["anonymous_id"] for r in private_rows} != ids:
        raise ValueError("public/private wave IDs do not match")

    public_result = {
        "schema_version": 1,
        "selection_rule": {"evaluation_wave": args.wave, "family": args.family},
        "selection_was_frozen_by_sha256_order_before_predictions": True,
        "outcome_fields_present": False,
        "records": rows,
    }
    private_result = {
        "schema_version": 1,
        "access": "private audit map; do not provide to predictors or annotators",
        "records": private_rows,
    }
    args.public_output.parent.mkdir(parents=True, exist_ok=True)
    args.private_output.parent.mkdir(parents=True, exist_ok=True)
    args.public_output.write_text(json.dumps(public_result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.private_output.write_text(json.dumps(private_result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"wave": args.wave, "family": args.family, "records": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
