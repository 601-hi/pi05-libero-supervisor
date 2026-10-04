"""Blind pilot for language-grounded boxes on frozen LIBERO RGB frames.

The model receives only an RGB image and goal language. Audit-only manifest
fields (suite, task id, outcome) are deliberately never read.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from vla_supervisor.goal_relations import (
    parse_goal_relation,
    source_entity_queries,
    visual_category_query,
    visual_entity_query,
)
from vla_supervisor.relation_candidate_selector import select_source_consistent_object


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="IDEA-Research/grounding-dino-tiny")
    parser.add_argument("--episodes-per-goal", type=int, default=1)
    parser.add_argument("--box-threshold", type=float, default=0.22)
    parser.add_argument("--text-threshold", type=float, default=0.18)
    parser.add_argument("--source-context", action="store_true")
    parser.add_argument("--first-frame-only", action="store_true")
    return parser.parse_args()


def choose_jobs(jobs: list[dict], episodes_per_goal: int) -> list[dict]:
    counts: dict[str, int] = {}
    selected = []
    for job in jobs:
        goal = str(job["goal_language"])
        if counts.get(goal, 0) >= episodes_per_goal:
            continue
        # Copy only inputs allowed by the feature firewall.
        selected.append({
            "sidecar": job["sidecar"],
            "goal_language": goal,
            "frame_indices": job["frame_indices"],
        })
        counts[goal] = counts.get(goal, 0) + 1
    return selected


def choose_frames(frame_indices: list[int], first_frame_only: bool = False) -> list[int]:
    if not frame_indices:
        return []
    if first_frame_only:
        return [frame_indices[0]]
    return sorted(set((frame_indices[0], frame_indices[len(frame_indices) // 2], frame_indices[-1])))


def draw_predictions(image: Image.Image, predictions: list[dict], caption: str) -> Image.Image:
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, canvas.width, 30), fill=(0, 0, 0))
    draw.text((4, 4), caption[:90], fill=(255, 255, 255))
    colors = ((255, 40, 40), (40, 255, 40), (40, 160, 255), (255, 220, 40))
    for index, item in enumerate(predictions):
        box = tuple(item["box_xyxy"])
        color = colors[index % len(colors)]
        draw.rectangle(box, outline=color, width=2)
        draw.text(
            (box[0] + 2, max(31, box[1] + 2)),
            f'{item.get("role", "entity")}:{item["label"]} {item["score"]:.2f}',
            fill=color,
        )
    return canvas


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    jobs = choose_jobs(manifest["jobs"], args.episodes_per_goal)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Keep Grounding DINO in FP32 for this correctness pilot. In the tested
    # Transformers build, forcing the complete model to FP16 leaves some text
    # features in FP32 and causes mixed-dtype failures in fusion layers.
    dtype = torch.float32
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(args.model, dtype=dtype).to(device)
    model.eval()

    records = []
    for job_index, job in enumerate(jobs):
        with np.load(job["sidecar"], mmap_mode="r") as sidecar:
            for frame_index in choose_frames(job["frame_indices"], args.first_frame_only):
                for view_key in ("agent_images", "wrist_images"):
                    if view_key not in sidecar:
                        continue
                    image = Image.fromarray(np.asarray(sidecar[view_key][frame_index], dtype=np.uint8)).convert("RGB")
                    predictions = []
                    relation = parse_goal_relation(job["goal_language"])
                    prompts = [("object", visual_entity_query(relation.manipulated_object))]
                    if relation.target and relation.relation != "close":
                        prompts.append(("target", visual_entity_query(relation.target)))
                    if args.source_context:
                        prompts[0] = ("object_category", visual_category_query(relation.manipulated_object))
                        prompts.extend(
                            (f"source_{index}", query)
                            for index, query in enumerate(source_entity_queries(relation))
                        )
                    for role, text_prompt in prompts:
                        inputs = processor(images=image, text=text_prompt + ".", return_tensors="pt").to(device)
                        inputs["pixel_values"] = inputs["pixel_values"].to(dtype=dtype)
                        with torch.inference_mode():
                            outputs = model(**inputs)
                        result = processor.post_process_grounded_object_detection(
                            outputs,
                            inputs.input_ids,
                            threshold=args.box_threshold,
                            text_threshold=args.text_threshold,
                            target_sizes=[image.size[::-1]],
                        )[0]
                        text_labels = result.get("text_labels", result["labels"])
                        for box, score, label in zip(result["boxes"], result["scores"], text_labels):
                            predictions.append({
                                "role": role,
                                "label": str(label),
                                "score": float(score),
                                "box_xyxy": [float(value) for value in box],
                            })
                    stem = f"goal{job_index:02d}_frame{frame_index:04d}_{view_key}"
                    selection = select_source_consistent_object(
                        predictions, relation.source_relation, image.size[::-1]
                    )
                    selected_predictions = [
                        predictions[index]
                        for index in (selection.object_prediction_index, *selection.source_prediction_indices)
                        if index is not None
                    ]
                    display_predictions = selected_predictions if selected_predictions else predictions
                    draw_predictions(image, display_predictions, job["goal_language"]).save(args.output_dir / f"{stem}.png")
                    records.append({
                        "goal_index": job_index,
                        "goal_language": job["goal_language"],
                        "goal_relation": relation.relation,
                        "object_prompt": relation.manipulated_object,
                        "target_prompt": relation.target,
                        "source_relation": relation.source_relation,
                        "source_description": relation.source_description,
                        "frame_index": frame_index,
                        "view": view_key,
                        "predictions": predictions,
                        "source_consistent_selection": {
                            "object_prediction_index": selection.object_prediction_index,
                            "source_prediction_indices": list(selection.source_prediction_indices),
                            "cost": selection.cost if np.isfinite(selection.cost) else None,
                            "runner_up_gap": selection.runner_up_gap if np.isfinite(selection.runner_up_gap) else None,
                            "ambiguous": selection.ambiguous,
                        },
                    })

    summary = {
        "schema_version": 1,
        "model": args.model,
        "device": device,
        "selection": "one episode per unique goal; first/middle/last frozen frame; both views",
        "feature_firewall": ["RGB image", "goal_language"],
        "num_unique_goals": len(jobs),
        "num_images": len(records),
        "records": records,
    }
    (args.output_dir / "predictions.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps({key: summary[key] for key in ("model", "device", "num_unique_goals", "num_images")}))


if __name__ == "__main__":
    main()
