import numpy as np

from cross_suite_generalization.export_initial_candidate_masks_gpu import select_items


def item(box, area, iou, stability):
    return {
        "bbox": box,
        "area": area,
        "predicted_iou": iou,
        "stability_score": stability,
        "segmentation": np.zeros((224, 224), dtype=bool),
    }


def test_selection_matches_frozen_area_sort_border_and_nms_rules():
    records = [
        item([20, 80, 20, 20], 400, 0.8, 0.9),
        item([21, 81, 20, 20], 400, 0.9, 0.9),  # selected first; suppresses previous box
        item([100, 90, 12, 12], 144, 0.85, 0.9),
        item([60, 2, 10, 10], 100, 0.99, 0.99),  # center above frozen y gate
        item([130, 100, 80, 80], 6400, 0.99, 0.99),  # above 8% area
    ]
    selected = select_items(records, (224, 224), max_candidates=12)
    assert [row["bbox"] for row in selected] == [[21, 81, 20, 20], [100, 90, 12, 12]]


def test_selection_respects_max_candidates():
    records = [item([10 + 20 * index, 100, 8, 8], 110, 0.9 - index * 0.01, 0.9) for index in range(5)]
    assert len(select_items(records, (224, 224), max_candidates=3)) == 3
