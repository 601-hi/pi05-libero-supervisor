from cross_suite_generalization.compare_localization_proposals import iou


def test_iou_is_scale_free_and_rejects_disjoint_boxes():
    assert iou([0,0,1,1],[0,0,1,1]) == 1
    assert iou([0,0,2,2],[1,1,3,3]) == 1/7
    assert iou([0,0,1,1],[2,2,3,3]) == 0
    assert iou([0,0,0,0],[0,0,0,0]) == 0
