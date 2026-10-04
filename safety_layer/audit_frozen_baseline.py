"""Verify that frozen execution-supervisor artifacts have not changed."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--tools-root", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    paths = {
        "normal_expert": args.results_root / "conditional_normal_dynamics_expert_frozen_v2.pt",
        "abnormal_expert": args.results_root / "abnormal_dynamics_expert_frozen_candidate_v1.pt",
        "fusion": args.results_root / "two_expert_fusion_frozen_v1.json",
        "evaluation": args.results_root / "two_expert_seed21_final_evaluation_v1.json",
        "fusion_code": args.tools_root / "two_expert_fusion.py",
        "evaluation_code": args.tools_root / "evaluate_two_expert_fusion.py",
    }
    mismatches = []
    for name, path in paths.items():
        actual = sha256(path) if path.is_file() else None
        expected = manifest["sha256"][name]
        ok = actual == expected
        print(f"{name}: {'OK' if ok else 'MISMATCH'} expected={expected} actual={actual}")
        if not ok:
            mismatches.append(name)
    if mismatches:
        raise SystemExit("Frozen baseline audit failed: " + ", ".join(mismatches))


if __name__ == "__main__":
    main()
