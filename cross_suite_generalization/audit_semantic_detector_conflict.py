"""Audit semantic detector agreement without treating detector scores as probabilities."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def iou(a: list[float], b: list[float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return intersection / max(area_a + area_b - intersection, 1e-12)


def object_predictions(row: dict) -> list[dict]:
    return [p for p in row["predictions"] if p["role"].startswith("object")]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dino", type=Path, required=True)
    parser.add_argument("--florence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--agreement-iou", type=float, default=0.5)
    args = parser.parse_args()
    dino = json.loads(args.dino.read_text(encoding="utf-8"))["records"]
    florence = json.loads(args.florence.read_text(encoding="utf-8"))["records"]
    florence_by_goal = {row["goal_index"]: row for row in florence}
    records = []
    for drow in dino:
        if drow["view"] != "agent_images":
            continue
        frow = florence_by_goal[drow["goal_index"]]
        dp, fp = object_predictions(drow), object_predictions(frow)
        overlaps = [
            {"dino_index": di, "florence_index": fi, "iou": iou(d["box_xyxy"], f["box_xyxy"])}
            for di, d in enumerate(dp) for fi, f in enumerate(fp)
        ]
        best = max(overlaps, key=lambda row: row["iou"], default=None)
        agreeing_dino = sorted({row["dino_index"] for row in overlaps if row["iou"] >= args.agreement_iou})
        # Agreement is a proposal, not identity proof. Multiple agreeing DINO boxes remain ambiguous.
        if len(agreeing_dino) == 1:
            disposition = "single_cross_model_proposal_requires_physical_confirmation"
        elif len(agreeing_dino) > 1:
            disposition = "ambiguous_cross_model_proposals"
        else:
            disposition = "no_cross_model_agreement"
        records.append({
            "goal_index": drow["goal_index"],
            "frame_index": drow["frame_index"],
            "dino_object_candidates": len(dp),
            "florence_object_candidates": len(fp),
            "best_cross_model_iou": None if best is None else best["iou"],
            "agreeing_dino_indices": agreeing_dino,
            "disposition": disposition,
            "warning": "scores are detector confidences, not calibrated P(correct identity)",
        })
    result = {
        "schema_version": 1,
        "agreement_iou": args.agreement_iou,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(records, ensure_ascii=False))


if __name__ == "__main__":
    main()
