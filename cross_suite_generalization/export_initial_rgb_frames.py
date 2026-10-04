#!/usr/bin/env python3
"""Export only the initial RGB frame for blind mask-identity adjudication."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--private", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    private = {r["anonymous_id"]: r for r in json.loads(args.private.read_text(encoding="utf-8"))["records"]}
    predictions = json.loads(args.predictions.read_text(encoding="utf-8"))["records"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for row in predictions:
        identifier = row["anonymous_id"]
        with np.load(private[identifier]["original_sidecar"], mmap_mode="r") as data:
            frame = np.asarray(data["agent_images"][0], dtype=np.uint8)
        Image.fromarray(frame).save(args.output_dir / f"{identifier}_initial_rgb.jpg", quality=98)
    print(json.dumps({"episodes": len(predictions), "output_dir": str(args.output_dir)}))


if __name__ == "__main__":
    main()
