#!/usr/bin/env python3
"""Fit on normal transitions and export calibration scores without selecting on test."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from action_conditioned_visual_residual import fit, save, score


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train", type=Path, required=True)
    p.add_argument("--calibration", type=Path, required=True)
    p.add_argument("--model-out", type=Path, required=True)
    p.add_argument("--calibration-scores-out", type=Path, required=True)
    p.add_argument("--report-out", type=Path, required=True)
    p.add_argument("--projected-channels", type=int, default=16)
    p.add_argument("--ridge", type=float, default=10.0)
    args = p.parse_args()
    train = np.load(args.train, allow_pickle=False)
    calibration = np.load(args.calibration, allow_pickle=False)
    model = fit(train, args.projected_channels, args.ridge)
    values = score(model, calibration)
    save(args.model_out, model)
    np.savez_compressed(args.calibration_scores_out, **values,
                        episode_id=calibration["episode_id"], task_id=calibration["task_id"],
                        active=calibration["previous_active_any"], ambiguous=calibration["previous_ambiguous_any"])
    normal = np.isclose(calibration["scale"].astype(float), 1.0)
    report = {
        "method": "action-conditioned patch visual residual ridge baseline",
        "uses_anomaly_labels_for_fit": False,
        "uses_task_identity": False,
        "train_clean_rows": int(np.isclose(train["scale"].astype(float), 1.0).sum()),
        "calibration_rows": int(len(calibration["scale"])),
        "normal_residual_quantiles": {str(q): float(np.quantile(values["residual_score"][normal], q))
                                      for q in (0.5, 0.9, 0.95, 0.99)},
        "warning": "Threshold and persistence must be selected on calibration episode maxima; frozen test remains unopened."
    }
    args.report_out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
