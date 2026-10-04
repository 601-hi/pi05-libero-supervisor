"""Evaluate a pre-specified cross-view confirmation threshold."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def chosen_similarity(bridge: dict) -> float:
    chosen = int(bridge["chosen_wrist_id"])
    return float(next(row["appearance"] for row in bridge["candidate_scores"] if int(row["candidate_id"]) == chosen))


def summarise(rows: list[dict], *, total_episodes: int) -> dict:
    confirmed = [row for row in rows if row["confirmed"]]
    correct = [row for row in confirmed if row["correct"]]
    return {
        "evaluable": len(rows),
        "confirmed": len(confirmed),
        "correct_confirmed": len(correct),
        "confirmation_precision": len(correct) / len(confirmed) if confirmed else None,
        "coverage_all_episodes": len(confirmed) / total_episodes if total_episodes else None,
        "coverage_evaluable": len(confirmed) / len(rows) if rows else None,
        "unknown_rate_all_episodes": 1.0 - len(confirmed) / total_episodes if total_episodes else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bridge", type=Path, required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source = json.loads(args.bridge.read_text(encoding="utf-8"))
    pipeline_rows: list[dict] = []
    oracle_rows: list[dict] = []
    for record in source["records"]:
        identifier = record["anonymous_id"]
        pipeline = record.get("pipeline_bridge")
        if pipeline is not None:
            score = chosen_similarity(pipeline)
            pipeline_rows.append(
                {
                    "anonymous_id": identifier,
                    "similarity": score,
                    "confirmed": score >= args.threshold,
                    "fixed_correct": bool(record["predicted_fixed_correct"]),
                    "wrist_correct": bool(pipeline["chosen_correct"]),
                    "correct": bool(record["predicted_fixed_correct"] and pipeline["chosen_correct"]),
                }
            )
        oracle = record.get("oracle_fixed_bridge")
        if oracle is not None:
            score = chosen_similarity(oracle)
            oracle_rows.append(
                {
                    "anonymous_id": identifier,
                    "similarity": score,
                    "confirmed": score >= args.threshold,
                    "wrist_correct": bool(oracle["chosen_correct"]),
                    "correct": bool(oracle["chosen_correct"]),
                }
            )

    total = len(source["records"])
    result = {
        "schema_version": 1,
        "threshold": args.threshold,
        "correctness_definition": "pipeline confirmation is correct only if both fixed target selection and wrist identity are correct",
        "pipeline": {"summary": summarise(pipeline_rows, total_episodes=total), "records": pipeline_rows},
        "oracle_fixed": {"summary": summarise(oracle_rows, total_episodes=total), "records": oracle_rows},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value["summary"] for key, value in result.items() if key in {"pipeline", "oracle_fixed"}}, indent=2))


if __name__ == "__main__":
    main()
