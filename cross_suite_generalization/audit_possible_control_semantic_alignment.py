"""Audit whether physical and semantic evidence land on the same fixed-view candidate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def xywh_to_xyxy(box):
    x, y, w, h = map(float, box)
    return [x, y, x + w, y + h]


def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    aa, ab = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1]), max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / max(aa + ab - inter, 1e-12)


def objects(row):
    return [p for p in row["predictions"] if p["role"].startswith("object")]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixed-predictions", type=Path, required=True)
    parser.add_argument("--fixed-motion", type=Path, required=True)
    parser.add_argument("--changepoint", type=Path, required=True)
    parser.add_argument("--fusion", type=Path, required=True)
    parser.add_argument("--dino", type=Path, required=True)
    parser.add_argument("--florence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pred = {r["event_id"]: r for r in json.loads(args.fixed_predictions.read_text())["records"]}
    motion = {r["event_id"]: r for r in json.loads(args.fixed_motion.read_text())["records"]}
    change = {r["event_id"]: r for r in json.loads(args.changepoint.read_text())["records"]}
    possible = [r for r in json.loads(args.fusion.read_text())["records"] if r["control_evidence_state"] == "possible_new_control"]
    dino_rows = {(r["frame_index"], r["view"]): r for r in json.loads(args.dino.read_text())["records"]}
    florence_by_goal = {r["goal_language"]: r for r in json.loads(args.florence.read_text())["records"]}
    output = []
    for event in possible:
        eid, frame = event["event_id"], int(event["close_frame"])
        drow = dino_rows[(frame, "agent_images")]
        frow = florence_by_goal[drow["goal_language"]]
        dboxes, fboxes = [p["box_xyxy"] for p in objects(drow)], [p["box_xyxy"] for p in objects(frow)]
        sustained = set(motion[eid]["sustained_candidate_ids"])
        changes = {r["candidate_id"]: r["combined_change"] for r in change[eid]["candidates"]}
        rows = []
        for candidate_id in sorted(sustained):
            box = xywh_to_xyxy(pred[eid]["boxes_xywh"][int(candidate_id) - 1])
            rows.append({
                "candidate_id": candidate_id,
                "combined_change": changes[candidate_id],
                "max_dino_object_iou": max((iou(box, value) for value in dboxes), default=0.0),
                "max_florence_object_iou": max((iou(box, value) for value in fboxes), default=0.0),
            })
        # IoU 0.3 is an audit display rule only, not a calibrated deployment gate.
        aligned = [r for r in rows if r["combined_change"] > 0 and r["max_dino_object_iou"] >= .3 and r["max_florence_object_iou"] >= .3]
        output.append({
            "event_id": eid, "goal_language": drow["goal_language"], "candidates": rows,
            "same_candidate_alignment_count": len(aligned),
            "disposition": "semantic_physical_alignment_candidate" if aligned else "abstain_evidence_split_across_candidates",
        })
    result = {
        "schema_version": 1,
        "warning": "IoU rule is a descriptive audit, not calibrated probability or deployment threshold",
        "records": output,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
