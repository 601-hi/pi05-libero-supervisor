#!/usr/bin/env python3
"""Compare deterministic local file ranges byte-for-byte with an HTTP object."""
from __future__ import annotations

import argparse
from pathlib import Path
import urllib.request


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("file", type=Path)
    args = parser.parse_args()
    size = args.file.stat().st_size
    block = 65536
    offsets = sorted(set((0, 24 * 1048576 - block // 2, 24 * 1048576, size // 2, size - block)))
    with args.file.open("rb") as local:
        for offset in offsets:
            end = min(size, offset + block) - 1
            request = urllib.request.Request(args.url, headers={"Range": f"bytes={offset}-{end}"})
            with urllib.request.urlopen(request, timeout=60) as remote:
                expected = remote.read()
            local.seek(offset)
            actual = local.read(len(expected))
            if actual != expected:
                raise RuntimeError(f"byte mismatch at range {offset}-{end}")
            print(f"MATCH {offset}-{end}")


if __name__ == "__main__":
    main()
