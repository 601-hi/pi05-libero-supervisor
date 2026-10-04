#!/usr/bin/env python3
"""Create an anonymous hard-link view of RGB sidecars for blinded tooling."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--public-manifest", type=Path, required=True)
    parser.add_argument("--private-map", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--index-output", type=Path, required=True)
    args = parser.parse_args()
    public = json.loads(args.public_manifest.read_text(encoding="utf-8"))["records"]
    private = {
        record["anonymous_id"]: record["original_sidecar"]
        for record in json.loads(args.private_map.read_text(encoding="utf-8"))["records"]
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for item in public:
        identifier = item["anonymous_id"]
        if identifier not in private:
            raise KeyError(f"missing private mapping for {identifier}")
        source = Path(private[identifier])
        destination = args.output_dir / f"{identifier}.npz"
        if destination.exists():
            if not os.path.samefile(source, destination):
                raise FileExistsError(f"refusing to replace nonmatching file: {destination}")
        else:
            os.link(source, destination)
        records.append({
            "anonymous_id": identifier,
            "sidecar": str(destination),
            "frame_count": item["frame_count"],
        })
    output = {
        "schema_version": 1,
        "access": "blinded RGB/action-index sidecar view",
        "excluded_metadata": ["task language", "outcome", "reward", "suite", "task id", "episode id"],
        "records": records,
    }
    args.index_output.parent.mkdir(parents=True, exist_ok=True)
    args.index_output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"anonymous_sidecars": len(records), "output_dir": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
