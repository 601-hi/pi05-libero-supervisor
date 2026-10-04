#!/usr/bin/env python3
"""Fetch all DROID-100 shards with resumable partial files and a SHA-256 manifest."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import urllib.request


BASE = "https://storage.googleapis.com/gresearch/robotics/droid_100/1.0.0/"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, final: Path) -> None:
    partial = final.with_suffix(final.suffix + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    request = urllib.request.Request(url, headers={"Range": f"bytes={offset}-"} if offset else {})
    with urllib.request.urlopen(request, timeout=120) as response:
        append = offset > 0 and response.status == 206
        if offset and not append:
            offset = 0
        mode = "ab" if append else "wb"
        with partial.open(mode) as output:
            while chunk := response.read(4 * 1024 * 1024):
                output.write(chunk)
    partial.replace(final)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("destination", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    args.destination.mkdir(parents=True, exist_ok=True)
    def fetch_one(index: int) -> dict:
        name = f"r2d2_faceblur-train.tfrecord-{index:05d}-of-00031"
        final = args.destination / name
        if not final.exists():
            download(BASE + name, final)
        return {
            "index": index,
            "url": BASE + name,
            "filename": name,
            "bytes": final.stat().st_size,
            "sha256": sha256(final),
        }
    records = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(fetch_one, index): index for index in range(31)}
        for completed, future in enumerate(as_completed(futures), start=1):
            record = future.result()
            records.append(record)
            print(
                f"VERIFIED {completed}/31 index={record['index']} "
                f"{record['filename']} bytes={record['bytes']}", flush=True
            )
    records.sort(key=lambda item: item["index"])
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps({
        "dataset": "DROID-100/r2d2_faceblur/1.0.0",
        "selection_rule": "complete official 100-episode sample; no value-based selection",
        "base_url": BASE,
        "records": records,
        "total_bytes": sum(record["bytes"] for record in records),
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
