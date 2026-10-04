#!/usr/bin/env python3
"""Apply the reconstructed frozen legacy-v2 MLP to natural outcome traces.

Only LIBERO task IDs 0--9 are compatible with this historical model.  Outcomes
are used for reporting only; the model, per-task threshold, and 2-of-3 rule are
reconstructed from the original training/calibration code.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-code-root", type=Path, required=True)
    parser.add_argument("--trace", type=Path, action="append", required=True)
    parser.add_argument("--reference-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.legacy_code_root))
    import evaluate_supervisor as ev
    import evaluate_v2_seed14_frozen as frozen
    import train_hybrid_supervisor_v2 as v2

    model, cov, predict, thresholds, loss = frozen.fit_frozen()
    reference = json.loads(args.reference_report.read_text(encoding="utf-8"))
    expected = {int(k): float(v) for k, v in reference["frozen_rule"]["thresholds"].items()}
    threshold_error = max(abs(thresholds[k] - expected[k]) for k in thresholds)
    details = []
    for file_index, path in enumerate(args.trace):
        rows, (records, _, x, y, _) = v2.load(str(path))
        unsupported = sorted({int(r["task_id"]) for r in records} - set(thresholds))
        if unsupported:
            raise ValueError(f"legacy v2 has no task thresholds for {unsupported}")
        features = v2.physical_features(model, cov, records, x, y)
        probability = predict(features)
        normalized = np.asarray([probability[i] / (thresholds[int(r["task_id"])] + 1e-12)
                                 for i, r in enumerate(records)])
        alarm = ev.persistence_series(normalized, records, 1.0, required=2, window=3)
        outcomes = {}
        for row in rows:
            if row.get("event") == "episode_end":
                outcomes[(int(row["task_id"]), int(row["episode_idx"]))] = bool(row.get("success", False))
        for (task_id, episode_idx), indices in ev.episode_groups(records).items():
            indices = sorted(indices, key=lambda i: int(records[i]["action_index"]))
            hits = [i for i in indices if alarm[i]]
            details.append({"file_index": file_index, "trace": str(path), "task_id": int(task_id),
                            "episode_idx": int(episode_idx), "success": outcomes[(task_id, episode_idx)],
                            "steps": len(indices), "alarm": bool(hits),
                            "first_alarm_action_index": int(records[hits[0]]["action_index"]) if hits else None,
                            "max_probability": float(max(probability[i] for i in indices)),
                            "max_normalized_risk": float(max(normalized[i] for i in indices))})
    success = [x for x in details if x["success"]]
    failure = [x for x in details if not x["success"]]
    result = {"model": "legacy_v2_binary_mlp_reconstructed_architecture",
              "exact_frozen_weights_available": False,
              "reconstruction_threshold_max_error": threshold_error,
              "rule": "2of3", "target_used_for_training_or_threshold_selection": False,
              "train_loss": loss, "successful_episodes": len(success), "failed_episodes": len(failure),
              "success_alarm_rate": float(np.mean([x["alarm"] for x in success])) if success else None,
              "failure_detection_rate": float(np.mean([x["alarm"] for x in failure])) if failure else None,
              "details": details}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "details"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
