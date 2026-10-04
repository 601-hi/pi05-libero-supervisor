"""Freeze fixed transparent baselines from multi-event development data only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from analyze_multi_event_visual_increment import auc, episode_rows, fit, predict


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--tracks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    tracks = json.loads(args.tracks.read_text(encoding="utf-8"))
    rows = episode_rows(manifest, tracks)
    labels = np.asarray([row["failure"] for row in rows], dtype=int)
    specs = {
        "action_only": ["mean_log_response", "min_log_response"],
        "visual_only": ["mean_log_event_visual", "mean_visual_log_drop"],
        "action_visual_fusion": [
            "mean_log_response", "min_log_response",
            "mean_log_event_visual", "mean_visual_log_drop",
        ],
    }
    models = {}
    for name, fields in specs.items():
        x = np.asarray([[row[field] for field in fields] for row in rows], dtype=float)
        model = fit(x, labels)
        scores = predict(model, x)
        alarms = scores >= model["threshold"]
        models[name] = {
            "feature_names": fields,
            "mean": model["mean"].tolist(),
            "scale": model["scale"].tolist(),
            "intercept": float(model["parameters"][0]),
            "coefficients": model["parameters"][1:].tolist(),
            "threshold_success_q95": float(model["threshold"]),
            "development_auc": auc(scores, labels),
            "development_failure_detection": float(np.mean(alarms[labels == 1])),
            "development_success_alarm": float(np.mean(alarms[labels == 0])),
        }
    payload = {
        "schema_version": 2,
        "training_firewall": "seed36 development only; fixed features and L2=1/N",
        "window_design": "two temporally separated events plus phase/command-matched controls",
        "episodes": len(rows),
        "failures": int(labels.sum()),
        "successes": int((1 - labels).sum()),
        "models": models,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({name: {
        "auc": model["development_auc"],
        "detection": model["development_failure_detection"],
        "success_alarm": model["development_success_alarm"],
    } for name, model in models.items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
