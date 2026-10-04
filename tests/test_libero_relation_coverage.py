from scripts.build_libero_relation_coverage import target_subtype


def test_all_raw_predicates_map_to_reusable_adapters():
    cases = [
        ("In", ["bowl", "cabinet_bottom_region"], "containment/drawer_interior"),
        ("In", ["mug", "microwave_1_heating_region"], "containment/microwave_cavity"),
        ("In", ["bowl", "basket_1_contain_region"], "containment/container_interior"),
        ("On", ["pot", "stove_1_cook_region"], "support/appliance_surface"),
        ("On", ["bowl", "plate"], "support/object_or_stack"),
        ("On", ["mug", "table_left_region"], "support/workspace_region"),
        ("Open", ["cabinet_top_region"], "articulation/drawer"),
        ("Close", ["microwave_1"], "articulation/microwave_door"),
        ("Turnon", ["flat_stove_1"], "device/binary_state"),
        ("Turnoff", ["flat_stove_1"], "device/binary_state"),
    ]
    for predicate, arguments, expected in cases:
        assert target_subtype(predicate, arguments) == expected
