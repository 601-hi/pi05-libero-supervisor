from cross_suite_generalization.evaluate_wrist_physical_fusion import distance_to_box, rank_fraction


def test_distance_to_box_is_zero_inside_and_euclidean_outside() -> None:
    box = [10.0, 20.0, 30.0, 40.0]
    assert distance_to_box((20.0, 30.0), box) == 0.0
    assert distance_to_box((7.0, 16.0), box) == 5.0


def test_rank_fraction_has_fixed_endpoints() -> None:
    result = rank_fraction({1: 2.0, 2: 4.0, 3: 3.0})
    assert result[2] == 1.0
    assert result[3] == 0.5
    assert result[1] == 0.0
