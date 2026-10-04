#!/usr/bin/env python3
"""Extract selected video frames and tile them into an event montage with ffmpeg."""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--frames", type=int, nargs="+", required=True)
    parser.add_argument("--columns", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.video.is_file():
        raise FileNotFoundError(args.video)
    rows = (len(args.frames) + args.columns - 1) // args.columns
    expression = "+".join(f"eq(n\\,{frame})" for frame in args.frames)
    vf = f"select='{expression}',scale=320:-1,tile={args.columns}x{rows}:padding=4:margin=4"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-i", str(args.video),
        "-vf", vf, "-frames:v", "1", str(args.output),
    ], check=True)
    print(args.output)


if __name__ == "__main__":
    main()
