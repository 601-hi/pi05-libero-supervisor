#!/usr/bin/env python3
"""Fetch the two metadata files and the smallest one-episode DROID-100 shard."""
from __future__ import annotations

import base64
import hashlib
from pathlib import Path
import urllib.request


BASE = "https://storage.googleapis.com/gresearch/robotics/droid_100/1.0.0/"
OBJECTS = {
    "dataset_info.json": (760, "JOcTGdwePw+iymeLhrqavg=="),
    "features.json": (18665, "AUayWu2IHU/GMy61byaQpg=="),
    "r2d2_faceblur-train.tfrecord-00009-of-00031": (9541347, "96W0orSYoQGtvvafj5oLtw=="),
}


def fetch(destination: Path, name: str, expected_size: int, expected_md5: str) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    final = destination / name
    partial = destination / f"{name}.part"
    digest = hashlib.md5()
    size = 0
    with urllib.request.urlopen(BASE + name, timeout=60) as response, partial.open("wb") as output:
        while chunk := response.read(1024 * 1024):
            output.write(chunk)
            digest.update(chunk)
            size += len(chunk)
    actual_md5 = base64.b64encode(digest.digest()).decode("ascii")
    if size != expected_size or actual_md5 != expected_md5:
        partial.unlink(missing_ok=True)
        raise ValueError(
            f"identity mismatch for {name}: size={size}, md5={actual_md5}"
        )
    partial.replace(final)
    print(f"VERIFIED {name} bytes={size} md5_base64={actual_md5}")


def main() -> None:
    destination = Path("/root/gpufree-data/droid-schema-sample/1.0.0")
    for name, identity in OBJECTS.items():
        fetch(destination, name, *identity)


if __name__ == "__main__":
    main()
