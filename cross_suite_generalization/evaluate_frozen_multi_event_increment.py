"""Apply frozen multi-event baselines without refitting."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.special import expit

from analyze_multi_event_visual_increment import auc, episode_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--tracks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frozen = json.loads(args.frozen.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    tracks = json.loads(args.tracks.read_text(encoding="utf-8"))
    rows = episode_rows(manifest, tracks)
    labels = np.asarray([row["failure"] for row in rows], dtype=int)
    metrics = {}
    for name, model in frozen["models"].items():
        x = np.asarray([[row[field] for field in model["feature_names"]] for row in rows])
        z = (x - np.asarray(model["mean"])) / np.asarray(model["scale"])
        scores = expit(model["intercept"] + z @ np.asarray(model["coefficients"]))
        alarms = scores >= model["threshold_success_q95"]
        metrics[name] = {
            "auc": auc(scores, labels),
            "failure_detection": float(np.mean(alarms[labels == 1])) if np.any(labels == 1) else None,
            "success_alarm": float(np.mean(alarms[labels == 0])) if np.any(labels == 0) else None,
            "detections": int(alarms[labels == 1].sum()),
            "failures": int(labels.sum()),
            "success_alarms": int(alarms[labels == 0].sum()),
            "successes": int((labels == 0).sum()),
            "scores": [{
                "episode_id": row["episode_id"], "group": row["group"],
                "failure": bool(label), "score": float(score), "alarm": bool(alarm),
            } for row, label, score, alarm in zip(rows, labels, scores, alarms)],
        }
    payload = {
        "schema_version": 2,
        "evaluation_firewall": "frozen seed36 model applied to seed37 without refit",
        "episodes": len(rows), "failures": int(labels.sum()), "successes": int((1-labels).sum()),
        "metrics": metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({name: {key: value[key] for key in (
        "auc", "detections", "failures", "success_alarms", "successes"
    )} for name, value in metrics.items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
