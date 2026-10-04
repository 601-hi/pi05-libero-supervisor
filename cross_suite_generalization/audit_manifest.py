#!/usr/bin/env python3
"""Static leakage and consistency checks for a generated experiment manifest."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("manifest", type=Path)
    args = p.parse_args()
    data = json.loads(args.manifest.read_text(encoding="utf-8"))
    rows = data["commands"]
    traces = [r["trace"] for r in rows]
    if len(traces) != len(set(traces)):
        raise ValueError("duplicate trace outputs")
    heldout = {r["suite"] for r in rows if r["split"] == "heldout_suite_test"}
    fitting = {r["suite"] for r in rows if r["split"] in {"development", "calibration"}}
    overlap = heldout & fitting
    if overlap:
        raise ValueError(f"held-out suites leaked into fitting splits: {sorted(overlap)}")
    for row in rows:
        if row["condition"] == "normal" and "--args.disturbance-num-steps 0" not in row["command"]:
            raise ValueError(f"normal command injects disturbance: {row['trace']}")
        if row["split"] == "heldout_suite_test" and not row["do_not_read_before_freeze"]:
            raise ValueError("held-out command is not sealed")
    reported = sum(int(r["expected_episodes"]) for r in rows)
    if reported != int(data["expected_total_episodes"]):
        raise ValueError("episode total mismatch")
    print(json.dumps({
        "ok": True, "commands": len(rows), "expected_episodes": reported,
        "fitting_suites": sorted(fitting), "heldout_suites": sorted(heldout),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
