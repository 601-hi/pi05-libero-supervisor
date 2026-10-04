#!/usr/bin/env python3
"""Complete one trusted contiguous .part prefix using verified parallel HTTP ranges."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import urllib.request


def fetch_range(url: str, path: Path, start: int, end: int) -> None:
    request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(request, timeout=120) as response, path.open("wb") as output:
        if response.status != 206:
            raise RuntimeError(f"server ignored byte range {start}-{end}")
        while chunk := response.read(4 * 1024 * 1024):
            output.write(chunk)
    if path.stat().st_size != end - start + 1:
        raise RuntimeError(f"range length mismatch for {start}-{end}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("partial", type=Path)
    parser.add_argument("--total-bytes", type=int, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    prefix = args.partial.stat().st_size
    if not 0 <= prefix < args.total_bytes:
        raise ValueError("partial must be a strict prefix no larger than the target")
    remaining = args.total_bytes - prefix
    edges = [prefix + remaining * i // args.workers for i in range(args.workers + 1)]
    ranges = [(edges[i], edges[i + 1] - 1) for i in range(args.workers) if edges[i] < edges[i + 1]]
    segment_paths = [args.partial.with_name(f"{args.partial.name}.range-{start}-{end}") for start, end in ranges]
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(fetch_range, args.url, path, start, end)
            for path, (start, end) in zip(segment_paths, ranges)
        ]
        for future in futures:
            future.result()
    with args.partial.open("ab") as output:
        for path in segment_paths:
            with path.open("rb") as source:
                while chunk := source.read(4 * 1024 * 1024):
                    output.write(chunk)
            path.unlink()
    if args.partial.stat().st_size != args.total_bytes:
        raise RuntimeError("completed file length mismatch")
    final = args.partial.with_suffix("")
    args.partial.replace(final)
    print(f"COMPLETED {final} bytes={final.stat().st_size}")


if __name__ == "__main__":
    main()
