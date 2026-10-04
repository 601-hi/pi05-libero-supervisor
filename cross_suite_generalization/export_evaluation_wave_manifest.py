#!/usr/bin/env python3
"""Export a public, outcome-free manifest for one preregistered evaluation wave."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--public", type=Path, required=True)
    parser.add_argument("--wave", type=int, required=True)
    parser.add_argument("--family", default="rigid_object_transport")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    public = json.loads(args.public.read_text(encoding="utf-8"))
    rows = [
        {key: row[key] for key in ("anonymous_id", "family", "goal_language", "selection_digest", "evaluation_wave")}
        for row in public["records"]
        if row["evaluation_wave"] == args.wave and row["family"] == args.family
    ]
    result = {
        "schema_version": 1,
        "selection_rule": {"evaluation_wave": args.wave, "family": args.family},
        "outcome_fields_present": False,
        "records": rows,
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"wave": args.wave, "family": args.family, "records": len(rows)}))


if __name__ == "__main__":
    main()
