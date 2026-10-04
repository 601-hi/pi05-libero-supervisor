"""CPU-only appearance bridge from fixed-view masks to wrist-view masks.

Wave2 labels are used only to audit oracle/pipeline accuracy.  The descriptor
and its weights are explicit and are not fitted on the evaluation labels.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def normalized_hist(values: np.ndarray, bins: int, value_range: tuple[float, float]) -> np.ndarray:
    hist, _ = np.histogram(values, bins=bins, range=value_range)
    hist = hist.astype(float)
    return hist / max(hist.sum(), 1.0)


def descriptor(image: np.ndarray, mask: np.ndarray) -> dict[str, np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    pixels = image[mask].astype(float) / 255.0
    if not len(pixels):
        zero = np.zeros(16, dtype=float)
        return {"rgb": np.tile(zero, 3), "gray": zero, "gradient": zero}
    rgb = np.concatenate([normalized_hist(pixels[:, channel], 16, (0.0, 1.0)) for channel in range(3)])
    gray_image = np.mean(image.astype(float) / 255.0, axis=2)
    gray = normalized_hist(gray_image[mask], 16, (0.0, 1.0))
    gy, gx = np.gradient(gray_image)
    magnitude = np.sqrt(gx * gx + gy * gy)
    # Robustly cap image edges so a single mask boundary cannot dominate.
    gradient = normalized_hist(np.clip(magnitude[mask], 0.0, 0.5), 16, (0.0, 0.5))
    return {"rgb": rgb, "gray": gray, "gradient": gradient}


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.dot(left, right) / (np.linalg.norm(left) * np.linalg.norm(right) + 1e-12))


def similarity(left: dict[str, np.ndarray], right: dict[str, np.ndarray]) -> dict[str, float]:
    parts = {key: cosine(left[key], right[key]) for key in left}
    parts["combined"] = 0.45 * parts["rgb"] + 0.35 * parts["gray"] + 0.20 * parts["gradient"]
    return parts


def load_mask_stack(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        return np.asarray(data["masks"], dtype=bool)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--public", type=Path, required=True)
    parser.add_argument("--private", type=Path, required=True)
    parser.add_argument("--fixed-masks", type=Path, required=True)
    parser.add_argument("--wrist-dir", type=Path, required=True)
    parser.add_argument("--fixed-annotations", type=Path, required=True)
    parser.add_argument("--wrist-annotations", type=Path, required=True)
    parser.add_argument("--fixed-replay", type=Path, required=True)
    parser.add_argument("--fixed-proposals", type=Path)
    parser.add_argument("--fixed-features", type=Path)
    parser.add_argument(
        "--fixed-physical-ranker",
        choices=["post_close_max_displacement"],
        help="Optionally reduce each fixed proposal set using a label-free physical rule.",
    )
    parser.add_argument("--changepoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    public = json.loads(args.public.read_text(encoding="utf-8"))["records"]
    private = {r["anonymous_id"]: r for r in json.loads(args.private.read_text(encoding="utf-8"))["records"]}
    fixed_truth = {
        r["anonymous_id"]: set(r["acceptable_target_mask_ids"])
        for r in json.loads(args.fixed_annotations.read_text(encoding="utf-8"))["records"]
    }
    wrist_truth = {
        key: set(value["acceptable_candidate_ids"])
        for key, value in json.loads(args.wrist_annotations.read_text(encoding="utf-8"))["annotations"].items()
    }
    replay = {
        r["anonymous_id"]: (None if r["first_lock"] is None else int(r["first_lock"]["candidate_id"]))
        for r in json.loads(args.fixed_replay.read_text(encoding="utf-8"))["records"]
    }
    proposals = None
    if args.fixed_proposals:
        proposals = {
            r["anonymous_id"]: set(map(int, r["proposal_ids"]))
            for r in json.loads(args.fixed_proposals.read_text(encoding="utf-8"))["records"]
        }
    fixed_features = None
    if args.fixed_features:
        fixed_features = {
            r["anonymous_id"]: r["candidates"]
            for r in json.loads(args.fixed_features.read_text(encoding="utf-8"))["records"]
        }
    if args.fixed_physical_ranker and (proposals is None or fixed_features is None):
        parser.error("--fixed-physical-ranker requires --fixed-proposals and --fixed-features")
    change_rows = {
        (r["anonymous_id"], int(r["candidate_id"])): float(r["area_change_log_drop"])
        for r in json.loads(args.changepoint.read_text(encoding="utf-8"))["candidates"]
    }

    rows = []
    for public_row in public:
        identifier = public_row["anonymous_id"]
        exported_fixed_rgb = args.fixed_masks / f"{identifier}_initial_rgb.jpg"
        if exported_fixed_rgb.exists():
            fixed_image = np.asarray(Image.open(exported_fixed_rgb).convert("RGB"))
        else:
            with np.load(private[identifier]["original_sidecar"], allow_pickle=False) as data:
                fixed_image = np.asarray(data["agent_images"][0], dtype=np.uint8)
        wrist_image = np.asarray(Image.open(args.wrist_dir / f"{identifier}_close_rgb.jpg").convert("RGB"))
        fixed_masks = load_mask_stack(args.fixed_masks / f"{identifier}_initial_masks.npz")
        wrist_masks = load_mask_stack(args.wrist_dir / f"{identifier}_initial_masks.npz")
        fixed_descriptors = [descriptor(fixed_image, mask) for mask in fixed_masks]
        wrist_descriptors = [descriptor(wrist_image, mask) for mask in wrist_masks]

        pair_scores = {}
        for fixed_id, fixed_desc in enumerate(fixed_descriptors, 1):
            pair_scores[fixed_id] = {
                wrist_id: similarity(fixed_desc, wrist_desc)
                for wrist_id, wrist_desc in enumerate(wrist_descriptors, 1)
            }

        def rank_for_fixed_ids(fixed_ids: set[int]) -> dict:
            if not fixed_ids:
                return None
            candidate_rows = []
            for wrist_id in range(1, len(wrist_descriptors) + 1):
                best_fixed_id = max(
                    fixed_ids, key=lambda fixed_id: pair_scores[fixed_id][wrist_id]["combined"]
                )
                appearance = pair_scores[best_fixed_id][wrist_id]["combined"]
                change = change_rows[(identifier, wrist_id)]
                candidate_rows.append(
                    {
                        "candidate_id": wrist_id,
                        "matched_fixed_id": best_fixed_id,
                        "appearance": appearance,
                        "area_change_log_drop": change,
                    }
                )
            # Two-stage ranking: appearance supplies identity, change point breaks
            # near-ties (within 0.03) rather than overriding a different-looking object.
            best_appearance = max(row["appearance"] for row in candidate_rows)
            shortlist = [row for row in candidate_rows if row["appearance"] >= best_appearance - 0.03]
            chosen_row = max(shortlist, key=lambda row: row["area_change_log_drop"])
            chosen = chosen_row["candidate_id"]
            ordered = sorted(candidate_rows, key=lambda row: (row["candidate_id"] != chosen, -row["appearance"]))
            return {
                "chosen_wrist_id": chosen,
                "chosen_fixed_id": chosen_row["matched_fixed_id"],
                "shortlist_ids": [row["candidate_id"] for row in shortlist],
                "chosen_correct": chosen in wrist_truth[identifier],
                "chosen_pair_correct": (
                    chosen in wrist_truth[identifier] and chosen_row["matched_fixed_id"] in fixed_truth[identifier]
                ),
                "candidate_scores": candidate_rows,
                "ordered_ids": [row["candidate_id"] for row in ordered],
            }

        oracle = rank_for_fixed_ids(fixed_truth[identifier])
        predicted_fixed = replay[identifier]
        pipeline = None if predicted_fixed is None else rank_for_fixed_ids({predicted_fixed})
        set_pipeline = None if proposals is None else rank_for_fixed_ids(proposals[identifier])
        physical_fixed = None
        physical_pipeline = None
        if args.fixed_physical_ranker:
            def physical_score(candidate_id: int) -> float:
                events = fixed_features[identifier][str(candidate_id)].get("events", [])
                values = [event.get("post40_max_displacement") for event in events]
                values = [float(value) for value in values if value is not None]
                return max(values) if values else 0.0

            if proposals[identifier]:
                physical_fixed = max(
                    proposals[identifier], key=lambda candidate_id: (physical_score(candidate_id), -candidate_id)
                )
                physical_pipeline = rank_for_fixed_ids({physical_fixed})
        rows.append(
            {
                "anonymous_id": identifier,
                "goal_language": public_row["goal_language"],
                "fixed_truth_ids": sorted(fixed_truth[identifier]),
                "predicted_fixed_id": predicted_fixed,
                "predicted_fixed_correct": predicted_fixed in fixed_truth[identifier] if predicted_fixed else False,
                "oracle_fixed_bridge": oracle,
                "pipeline_bridge": pipeline,
                "fixed_proposal_ids": None if proposals is None else sorted(proposals[identifier]),
                "fixed_proposal_contains_truth": (
                    None if proposals is None else bool(proposals[identifier].intersection(fixed_truth[identifier]))
                ),
                "set_pipeline_bridge": set_pipeline,
                "physical_ranked_fixed_id": physical_fixed,
                "physical_ranked_fixed_correct": (
                    None if physical_fixed is None else physical_fixed in fixed_truth[identifier]
                ),
                "physical_ranked_pipeline_bridge": physical_pipeline,
            }
        )

    pipeline_rows = [r for r in rows if r["pipeline_bridge"] is not None]
    set_pipeline_rows = [r for r in rows if r["set_pipeline_bridge"] is not None]
    physical_rows = [r for r in rows if r["physical_ranked_pipeline_bridge"] is not None]
    result = {
        "schema_version": 1,
        "protocol": {
            "cpu_only": True,
            "descriptor": "masked RGB histogram + grayscale histogram + gradient-magnitude histogram",
            "appearance_weights": {"rgb": 0.45, "gray": 0.35, "gradient": 0.20},
            "shortlist_margin": 0.03,
            "tie_breaker": "largest frozen area-change log drop within appearance shortlist",
            "evaluation_role": "determined by the supplied manifest; no parameter fitting occurs in this evaluator",
        },
        "summary": {
            "episodes": len(rows),
            "fixed_truth_observable": sum(bool(r["fixed_truth_ids"]) for r in rows),
            "fixed_selector_correct": sum(r["predicted_fixed_correct"] for r in rows),
            "wrist_truth_observable": sum(bool(wrist_truth[r["anonymous_id"]]) for r in rows),
            "oracle_fixed_bridge_evaluable": sum(r["oracle_fixed_bridge"] is not None for r in rows),
            "oracle_fixed_bridge_correct": sum(
                r["oracle_fixed_bridge"] is not None and r["oracle_fixed_bridge"]["chosen_correct"] for r in rows
            ),
            "pipeline_available": len(pipeline_rows),
            "pipeline_bridge_correct": sum(r["pipeline_bridge"]["chosen_correct"] for r in pipeline_rows),
            "full_pipeline_both_fixed_and_wrist_correct": sum(
                r["predicted_fixed_correct"] and r["pipeline_bridge"] is not None and r["pipeline_bridge"]["chosen_correct"]
                for r in rows
            ),
            "set_pipeline_available": len(set_pipeline_rows) if proposals is not None else None,
            "fixed_proposal_contains_truth": (
                sum(bool(r["fixed_proposal_contains_truth"]) for r in rows) if proposals is not None else None
            ),
            "set_pipeline_pair_correct": (
                sum(r["set_pipeline_bridge"]["chosen_pair_correct"] for r in set_pipeline_rows)
                if proposals is not None else None
            ),
            "physical_ranked_fixed_correct": (
                sum(bool(r["physical_ranked_fixed_correct"]) for r in physical_rows)
                if args.fixed_physical_ranker else None
            ),
            "physical_ranked_pair_correct": (
                sum(r["physical_ranked_pipeline_bridge"]["chosen_pair_correct"] for r in physical_rows)
                if args.fixed_physical_ranker else None
            ),
        },
        "records": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
