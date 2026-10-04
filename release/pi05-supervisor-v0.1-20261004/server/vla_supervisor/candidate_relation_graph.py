"""Task-agnostic relations between overlapping segmentation candidates.

The graph deliberately uses only mask geometry.  It does not use simulator
object state, task outcome, or a candidate selected by the controller.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Mapping

import numpy as np


@dataclass(frozen=True)
class CandidateRelation:
    left_id: Hashable
    right_id: Hashable
    intersection_over_left: float
    intersection_over_right: float
    iou: float
    boundary_distance_px: float
    relation: str


@dataclass(frozen=True)
class CandidateRelationGraph:
    areas: Mapping[Hashable, int]
    relations: tuple[CandidateRelation, ...]

    def equivalents(self, candidate_id: Hashable) -> frozenset[Hashable]:
        result = {candidate_id}
        for edge in self.relations:
            if edge.relation != "equivalent":
                continue
            if edge.left_id == candidate_id:
                result.add(edge.right_id)
            elif edge.right_id == candidate_id:
                result.add(edge.left_id)
        return frozenset(result)

    def whole_candidates_for(self, candidate_id: Hashable) -> frozenset[Hashable]:
        """Return geometrically larger candidates that contain this part."""
        result: set[Hashable] = set()
        for edge in self.relations:
            if edge.left_id == candidate_id and edge.relation == "left_part_of_right":
                result.add(edge.right_id)
            elif edge.right_id == candidate_id and edge.relation == "right_part_of_left":
                result.add(edge.left_id)
        return frozenset(result)

    def adjacent_candidates(self, candidate_id: Hashable, maximum_distance_px: float = 3.0) -> frozenset[Hashable]:
        result: set[Hashable] = set()
        for edge in self.relations:
            if edge.boundary_distance_px > maximum_distance_px:
                continue
            if edge.left_id == candidate_id:
                result.add(edge.right_id)
            elif edge.right_id == candidate_id:
                result.add(edge.left_id)
        return frozenset(result)


def _as_mask(value: np.ndarray) -> np.ndarray:
    mask = np.asarray(value, dtype=bool)
    if mask.ndim != 2:
        raise ValueError(f"candidate mask must be 2D, got {mask.shape}")
    return mask


def _boundary(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, 1, constant_values=False)
    interior = mask.copy()
    for row_shift, column_shift in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        interior &= padded[1 + row_shift : 1 + row_shift + mask.shape[0], 1 + column_shift : 1 + column_shift + mask.shape[1]]
    return np.argwhere(mask & ~interior)


def _minimum_boundary_distance(left: np.ndarray, right: np.ndarray) -> float:
    if not len(left) or not len(right):
        return float("inf")
    best = float("inf")
    for start in range(0, len(left), 256):
        delta = left[start : start + 256, None, :] - right[None, :, :]
        best = min(best, float(np.sqrt(np.sum(delta * delta, axis=2).min())))
    return best


def build_candidate_relation_graph(
    masks: Mapping[Hashable, np.ndarray],
    *,
    equivalent_iou: float = 0.80,
    containment_fraction: float = 0.85,
    maximum_part_area_ratio: float = 0.75,
) -> CandidateRelationGraph:
    """Build symmetric equivalence and directed part/whole relations.

    A high overlap alone is not called ``part_of``: the contained candidate
    must also be materially smaller.  This prevents near-duplicate SAM masks
    from being mistaken for a semantic part hierarchy.
    """
    if not 0 <= equivalent_iou <= 1 or not 0 <= containment_fraction <= 1:
        raise ValueError("overlap thresholds must be in [0, 1]")
    if not 0 < maximum_part_area_ratio < 1:
        raise ValueError("maximum_part_area_ratio must be in (0, 1)")
    converted = {key: _as_mask(value) for key, value in masks.items()}
    shapes = {value.shape for value in converted.values()}
    if len(shapes) > 1:
        raise ValueError("all candidate masks must share shape")
    areas = {key: int(value.sum()) for key, value in converted.items()}
    boundaries = {key: _boundary(value) for key, value in converted.items()}
    relations: list[CandidateRelation] = []
    keys = list(converted)
    for left_index, left_id in enumerate(keys):
        for right_id in keys[left_index + 1 :]:
            left, right = converted[left_id], converted[right_id]
            intersection = int(np.logical_and(left, right).sum())
            union = int(np.logical_or(left, right).sum())
            left_fraction = intersection / max(areas[left_id], 1)
            right_fraction = intersection / max(areas[right_id], 1)
            iou = intersection / max(union, 1)
            boundary_distance = _minimum_boundary_distance(boundaries[left_id], boundaries[right_id])
            area_ratio = min(areas[left_id], areas[right_id]) / max(areas[left_id], areas[right_id], 1)
            relation = "overlap"
            if iou >= equivalent_iou:
                relation = "equivalent"
            elif left_fraction >= containment_fraction and area_ratio <= maximum_part_area_ratio:
                relation = "left_part_of_right"
            elif right_fraction >= containment_fraction and area_ratio <= maximum_part_area_ratio:
                relation = "right_part_of_left"
            relations.append(CandidateRelation(left_id, right_id, left_fraction, right_fraction, iou, boundary_distance, relation))
    return CandidateRelationGraph(areas, tuple(relations))
