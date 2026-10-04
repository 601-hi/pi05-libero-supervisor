"""Development-only coverage/risk audit for cross-view confirmation gates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bridge", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    bridge = json.loads(args.bridge.read_text(encoding="utf-8"))["records"]
    replay = {r["anonymous_id"]: r for r in json.loads(args.replay.read_text(encoding="utf-8"))["records"]}
    rows = []
    for record in bridge:
        pipeline = record["pipeline_bridge"]
        first_lock = replay[record["anonymous_id"]]["first_lock"]
        if pipeline is None or first_lock is None:
            top_similarity = lock_margin = None
        else:
            top_similarity = max(row["appearance"] for row in pipeline["candidate_scores"])
            lock_frame = replay[record["anonymous_id"]]["frames"][first_lock["frame"]]
            lock_margin = float(lock_frame["score_margin"])
        rows.append(
            {
                "anonymous_id": record["anonymous_id"],
                "fixed_correct": record["predicted_fixed_correct"],
                "wrist_correct": pipeline is not None and pipeline["chosen_correct"],
                "full_correct": record["predicted_fixed_correct"] and pipeline is not None and pipeline["chosen_correct"],
                "top_similarity": top_similarity,
                "lock_margin": lock_margin,
            }
        )

    gates = []
    for similarity_threshold in [0.80, 0.82, 0.84, 0.86, 0.88, 0.90]:
        for lock_threshold in [0.0, 0.015, 0.020, 0.025, 0.030]:
            accepted = [
                r for r in rows
                if r["top_similarity"] is not None
                and r["top_similarity"] >= similarity_threshold
                and r["lock_margin"] >= lock_threshold
            ]
            gates.append(
                {
                    "similarity_threshold": similarity_threshold,
                    "lock_margin_threshold": lock_threshold,
                    "accepted": len(accepted),
                    "unknown": len(rows) - len(accepted),
                    "coverage": len(accepted) / len(rows),
                    "fixed_precision_among_accepted": (
                        sum(r["fixed_correct"] for r in accepted) / len(accepted) if accepted else None
                    ),
                    "full_precision_among_accepted": (
                        sum(r["full_correct"] for r in accepted) / len(accepted) if accepted else None
                    ),
                    "accepted_full_correct": sum(r["full_correct"] for r in accepted),
                    "accepted_full_wrong": sum(not r["full_correct"] for r in accepted),
                }
            )
    result = {
        "schema_version": 1,
        "interpretation": "Wave2 development sweep. Choose/freeze a gate before Wave3; do not quote the best row as test performance.",
        "rows": rows,
        "gates": gates,
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for gate in gates:
        if gate["accepted"] >= 8 and gate["accepted_full_wrong"] <= 1:
            print(json.dumps(gate))


if __name__ == "__main__":
    main()
