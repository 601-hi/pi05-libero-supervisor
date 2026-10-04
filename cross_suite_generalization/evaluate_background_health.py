"""Report target-agnostic health statistics for fixed-view background estimation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def finite(values: list[object]) -> np.ndarray:
    return np.asarray([float(value) for value in values if value is not None and np.isfinite(float(value))])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = json.loads(args.features.read_text(encoding="utf-8"))["records"]
    episode_rows = []
    all_confidence = []
    for record in records:
        confidence = finite(record["background_confidence_sequence"][1:])
        all_confidence.extend(confidence.tolist())
        episode_rows.append(
            {
                "anonymous_id": record["anonymous_id"],
                "background_valid_fraction": float(record["background_valid_fraction"]),
                "confidence_median": float(np.median(confidence)),
                "confidence_p10": float(np.percentile(confidence, 10)),
            }
        )
    values = np.asarray(all_confidence, dtype=float)
    result = {
        "schema_version": 1,
        "target_agnostic": True,
        "interpretation_limit": "Internal availability and confidence of background estimation; not independent-motion recall or target identity accuracy.",
        "summary": {
            "episodes": len(records),
            "episodes_with_nonzero_valid_fraction": sum(row["background_valid_fraction"] > 0 for row in episode_rows),
            "valid_fraction_mean": float(np.mean([row["background_valid_fraction"] for row in episode_rows])),
            "valid_fraction_min": float(np.min([row["background_valid_fraction"] for row in episode_rows])),
            "evaluated_frames_excluding_initial": len(values),
            "confidence_p10": float(np.percentile(values, 10)),
            "confidence_median": float(np.median(values)),
            "confidence_p90": float(np.percentile(values, 90)),
            "fraction_confidence_ge_0p5": float(np.mean(values >= 0.5)),
            "fraction_confidence_ge_0p3": float(np.mean(values >= 0.3)),
        },
        "episodes": episode_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
