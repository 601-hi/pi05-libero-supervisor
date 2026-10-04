#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from visual_motion_response import fit, save, score


def auc(labels, values):
    positive, negative = values[labels], values[~labels]
    return float(((positive[:, None] > negative[None, :]).sum() +
                  .5 * (positive[:, None] == negative[None, :]).sum()) /
                 (len(positive) * len(negative)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", type=Path, required=True)
    p.add_argument("--calibration", type=Path, required=True)
    p.add_argument("--model-out", type=Path, required=True)
    p.add_argument("--scores-out", type=Path, required=True)
    p.add_argument("--report-out", type=Path, required=True)
    p.add_argument("--ridge", type=float, default=10.0)
    args = p.parse_args()
    train = np.load(args.train, allow_pickle=False)
    calibration = np.load(args.calibration, allow_pickle=False)
    model = fit(train, args.ridge)
    values = score(model, calibration)
    save(args.model_out, model)
    np.savez_compressed(args.scores_out, **values, valid=calibration["valid"],
                        abnormal=calibration["abnormal"], episode_id=calibration["episode_id"],
                        action_index=calibration["action_index"], scale=calibration["scale"])
    valid = calibration["valid"].astype(bool)
    abnormal = calibration["abnormal"].astype(bool)
    report = {
        "method": "task-agnostic command-conditioned 4x4 directional optical-flow response",
        "appearance_features_used": False, "task_identity_used": False,
        "anomaly_labels_used_for_fit": False,
        "valid_normal_rows": int((valid & ~abnormal).sum()),
        "valid_active_rows": int((valid & abnormal).sum()),
        "calibration_auc": {
            "high_residual": auc(abnormal[valid], values["residual_score"][valid]),
            "low_directed_response": auc(abnormal[valid], -values["directed_response"][valid]),
            "low_cosine": auc(abnormal[valid], -values["cosine"][valid]),
        },
        "warning": "Calibration diagnostic only. No frozen-test data read and no threshold frozen yet."
    }
    args.report_out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
