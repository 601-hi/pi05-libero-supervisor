"""Blind Florence-2 phrase-grounding pilot on frozen first frames."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from transformers import AutoModelForCausalLM, AutoProcessor

from vla_supervisor.goal_relations import (
    parse_goal_relation,
    source_entity_queries,
    visual_category_query,
    visual_entity_query,
)
from vla_supervisor.relation_candidate_selector import select_source_consistent_object


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="microsoft/Florence-2-base-ft")
    return parser.parse_args()


def unique_goal_jobs(jobs: list[dict]) -> list[dict]:
    chosen = {}
    for job in jobs:
        goal = str(job["goal_language"])
        chosen.setdefault(goal, {
            "sidecar": job["sidecar"],
            "goal_language": goal,
            "frame_index": job["frame_indices"][0],
        })
    return list(chosen.values())


def main() -> None:
    args = arguments()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    jobs = unique_goal_jobs(manifest["jobs"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, dtype=dtype, attn_implementation="eager"
    ).to(device).eval()
    task = "<CAPTION_TO_PHRASE_GROUNDING>"
    records = []
    for goal_index, job in enumerate(jobs):
        relation = parse_goal_relation(job["goal_language"])
        prompts = [
            ("object_exact", visual_entity_query(relation.manipulated_object)),
            ("object_category", visual_category_query(relation.manipulated_object)),
        ]
        if relation.target and relation.relation != "close":
            prompts.append(("target", visual_entity_query(relation.target)))
        prompts.extend(
            (f"source_{index}", query)
            for index, query in enumerate(source_entity_queries(relation))
        )
        with np.load(job["sidecar"], mmap_mode="r") as sidecar:
            image = Image.fromarray(
                np.asarray(sidecar["agent_images"][job["frame_index"]], dtype=np.uint8)
            ).convert("RGB")
        predictions = []
        for role, phrase in prompts:
            inputs = processor(text=task + phrase, images=image, return_tensors="pt")
            input_ids = inputs["input_ids"].to(device)
            pixel_values = inputs["pixel_values"].to(device=device, dtype=dtype)
            with torch.inference_mode():
                generated_ids = model.generate(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    max_new_tokens=128,
                    num_beams=3,
                    do_sample=False,
                    use_cache=False,
                )
            generated = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
            parsed = processor.post_process_generation(generated, task=task, image_size=image.size)[task]
            for box, label in zip(parsed.get("bboxes", []), parsed.get("labels", [])):
                predictions.append({
                    "role": role,
                    "query": phrase,
                    "label": str(label),
                    # Florence phrase grounding does not expose calibrated box
                    # confidence. A neutral constant lets geometry rank boxes
                    # without pretending that a probability is available.
                    "score": 1.0,
                    "box_xyxy": [float(value) for value in box],
                })
        selection = select_source_consistent_object(predictions, relation.source_relation, image.size[::-1])
        canvas = image.copy()
        draw = ImageDraw.Draw(canvas)
        draw.rectangle((0, 0, canvas.width, 28), fill=(0, 0, 0))
        draw.text((3, 3), job["goal_language"][:90], fill=(255, 255, 255))
        colors = {"object_exact": (255, 40, 40), "object_category": (40, 255, 40), "target": (40, 160, 255)}
        for item in predictions:
            box = item["box_xyxy"]
            color = colors.get(item["role"], (180, 80, 255))
            draw.rectangle(box, outline=color, width=2)
            draw.text((box[0] + 2, max(30, box[1] + 2)), item["role"], fill=color)
        canvas.save(args.output_dir / f"goal{goal_index:02d}_agent.png")
        records.append({
            "goal_index": goal_index,
            "goal_language": job["goal_language"],
            "goal_relation": relation.relation,
            "predictions": predictions,
            "source_consistent_selection": {
                "object_prediction_index": selection.object_prediction_index,
                "source_prediction_indices": list(selection.source_prediction_indices),
                "cost": selection.cost if np.isfinite(selection.cost) else None,
                "runner_up_gap": selection.runner_up_gap if np.isfinite(selection.runner_up_gap) else None,
                "ambiguous": selection.ambiguous,
            },
        })
    result = {
        "schema_version": 1,
        "model": args.model,
        "device": device,
        "feature_firewall": ["agent RGB first frame", "goal_language"],
        "records": records,
    }
    (args.output_dir / "predictions.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps({"model": args.model, "device": device, "images": len(records)}))


if __name__ == "__main__":
    main()
