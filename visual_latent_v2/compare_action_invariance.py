"""Require exact action equality between latent-off and latent-on paired traces."""

import argparse
import json
from pathlib import Path

import numpy as np


def load(path: Path, event: str):
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("event") == event:
                rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--off", type=Path, required=True)
    parser.add_argument("--on", type=Path, required=True)
    args = parser.parse_args()
    off_inf, on_inf = load(args.off, "inference"), load(args.on, "inference")
    off_step, on_step = load(args.off, "step"), load(args.on, "step")
    if len(off_inf) != len(on_inf) or len(off_step) != len(on_step):
        raise AssertionError("Trace lengths differ; monitoring is not action-invariant")
    chunk_max = max(
        (float(np.max(np.abs(np.asarray(a["action_chunk"]) - np.asarray(b["action_chunk"])))) for a, b in zip(off_inf, on_inf)),
        default=0.0,
    )
    step_max = max(
        (float(np.max(np.abs(np.asarray(a["intended_action"]) - np.asarray(b["intended_action"])))) for a, b in zip(off_step, on_step)),
        default=0.0,
    )
    eef_max = max(
        (float(np.max(np.abs(np.asarray(a["eef_pos_after"]) - np.asarray(b["eef_pos_after"])))) for a, b in zip(off_step, on_step)),
        default=0.0,
    )
    result = {"inferences": len(off_inf), "steps": len(off_step), "chunk_max_abs": chunk_max, "step_max_abs": step_max, "eef_max_abs_m": eef_max}
    print(json.dumps(result, ensure_ascii=False))
    if chunk_max != 0.0 or step_max != 0.0 or eef_max != 0.0:
        raise AssertionError("Visual-latent side path changed the paired rollout")


if __name__ == "__main__":
    main()
