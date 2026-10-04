#!/usr/bin/env python3
"""Fetch preregistered DROID shards and write a content-identity manifest."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import urllib.request


BASE = "https://storage.googleapis.com/gresearch/robotics/droid_100/1.0.0/"
# Chosen before inspecting trajectory values: early, middle, and late indices;
# all are small shards, keeping this schema/alignment pilot inexpensive.
INDICES = (6, 9, 19, 29)


def main() -> None:
    destination = Path("/root/gpufree-data/droid-schema-sample/1.0.0")
    destination.mkdir(parents=True, exist_ok=True)
    records = []
    for index in INDICES:
        name = f"r2d2_faceblur-train.tfrecord-{index:05d}-of-00031"
        final, partial = destination / name, destination / f"{name}.part"
        if not final.exists():
            digest, size = hashlib.sha256(), 0
            with urllib.request.urlopen(BASE + name, timeout=90) as response, partial.open("wb") as output:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)
            partial.replace(final)
        digest = hashlib.sha256()
        with final.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
        record = {
            "index": index, "url": BASE + name, "filename": name,
            "bytes": final.stat().st_size, "sha256": digest.hexdigest(),
        }
        records.append(record)
        print(json.dumps(record))
    manifest = destination.parent / "preregistered_shards_manifest.json"
    manifest.write_text(json.dumps({
        "selection_rule": "indices 6,9,19,29 selected before value inspection; span early/middle/late and small shards",
        "records": records,
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
