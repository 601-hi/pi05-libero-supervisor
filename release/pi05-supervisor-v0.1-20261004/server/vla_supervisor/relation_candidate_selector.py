"""Rank grounded object boxes by language-specified source geometry."""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SourceConsistentSelection:
    object_prediction_index: int | None
    source_prediction_indices: tuple[int, ...]
    cost: float
    runner_up_gap: float

    @property
    def ambiguous(self) -> bool:
        return self.object_prediction_index is None or self.runner_up_gap < 0.05


def _box(item: dict) -> np.ndarray:
    box = np.asarray(item["box_xyxy"], dtype=float)
    if box.shape != (4,) or not np.all(np.isfinite(box)):
        raise ValueError("box_xyxy must contain four finite values")
    return box


def _center(box: np.ndarray) -> np.ndarray:
    return np.array(((box[0] + box[2]) / 2, (box[1] + box[3]) / 2))


def _area(box: np.ndarray) -> float:
    return float(max(0, box[2] - box[0]) * max(0, box[3] - box[1]))


def _intersection_over_object(obj: np.ndarray, ref: np.ndarray) -> float:
    width = max(0, min(obj[2], ref[2]) - max(obj[0], ref[0]))
    height = max(0, min(obj[3], ref[3]) - max(obj[1], ref[1]))
    return float(width * height / max(_area(obj), 1e-9))


def _center_inside(obj: np.ndarray, ref: np.ndarray) -> bool:
    point = _center(obj)
    return bool(ref[0] <= point[0] <= ref[2] and ref[1] <= point[1] <= ref[3])


def _point_segment_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    delta = end - start
    fraction = float(np.dot(point - start, delta) / max(np.dot(delta, delta), 1e-9))
    projection = start + np.clip(fraction, 0.0, 1.0) * delta
    return float(np.linalg.norm(point - projection))


def _pair_cost(obj: dict, ref: dict, relation: str, diagonal: float) -> float:
    obj_box, ref_box = _box(obj), _box(ref)
    distance = float(np.linalg.norm(_center(obj_box) - _center(ref_box)) / diagonal)
    overlap = _intersection_over_object(obj_box, ref_box)
    same_entity_penalty = overlap
    confidence_bonus = 0.08 * (float(obj["score"]) + float(ref["score"]))
    if relation == "in":
        geometry = (0.0 if _center_inside(obj_box, ref_box) else distance + 0.5) + (1.0 - overlap)
    elif relation == "next_to":
        geometry = distance + 1.5 * same_entity_penalty
    else:  # on
        area_ratio = _area(obj_box) / max(_area(ref_box), 1e-9)
        geometry = distance + 1.5 * same_entity_penalty + max(0.0, area_ratio - 0.8)
    return geometry - confidence_bonus


def select_source_consistent_object(
    predictions: list[dict], source_relation: str | None, image_shape: tuple[int, int]
) -> SourceConsistentSelection:
    if source_relation not in {"on", "in", "next_to", "between"}:
        return SourceConsistentSelection(None, (), float("inf"), 0.0)
    objects = [(index, item) for index, item in enumerate(predictions) if item.get("role") == "object_category"]
    source_roles = sorted({str(item.get("role")) for item in predictions if str(item.get("role", "")).startswith("source_")})
    source_groups = [[(index, item) for index, item in enumerate(predictions) if item.get("role") == role] for role in source_roles]
    required_sources = 2 if source_relation == "between" else 1
    if not objects or len(source_groups) < required_sources or any(not group for group in source_groups[:required_sources]):
        return SourceConsistentSelection(None, (), float("inf"), 0.0)
    diagonal = float(np.hypot(*image_shape))
    candidates: list[tuple[float, int, tuple[int, ...]]] = []
    for object_index, obj in objects:
        for sources in itertools.product(*source_groups[:required_sources]):
            source_indices = tuple(index for index, _ in sources)
            if source_relation == "between":
                obj_box = _box(obj)
                left, right = (_box(item) for _, item in sources)
                line_distance = _point_segment_distance(_center(obj_box), _center(left), _center(right)) / diagonal
                midpoint_distance = np.linalg.norm(_center(obj_box) - (_center(left) + _center(right)) / 2) / diagonal
                overlap = max(_intersection_over_object(obj_box, left), _intersection_over_object(obj_box, right))
                confidence_bonus = 0.08 * (float(obj["score"]) + sum(float(item["score"]) for _, item in sources))
                cost = float(line_distance + 0.5 * midpoint_distance + 1.5 * overlap - confidence_bonus)
            else:
                cost = _pair_cost(obj, sources[0][1], source_relation, diagonal)
            candidates.append((cost, object_index, source_indices))
    candidates.sort(key=lambda item: item[0])
    best = candidates[0]
    runner_up_gap = float(candidates[1][0] - best[0]) if len(candidates) > 1 else float("inf")
    return SourceConsistentSelection(best[1], best[2], float(best[0]), runner_up_gap)
