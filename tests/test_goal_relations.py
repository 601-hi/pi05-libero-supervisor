from vla_supervisor.goal_relations import (
    parse_goal_relation,
    source_entity_queries,
    visual_category_query,
    visual_entity_query,
)


def test_pick_place_ignores_source_relation_and_extracts_destination():
    result = parse_goal_relation(
        "pick up the black bowl between the plate and the ramekin and place it on the plate"
    )
    assert result.relation == "place_on"
    assert result.manipulated_object == "the black bowl"
    assert result.target == "the plate"
    assert result.source_relation == "between"
    assert source_entity_queries(result) == ("plate", "ramekin")


def test_put_in_relation():
    result = parse_goal_relation("put the black bowl in the top drawer of the cabinet")
    assert result.relation == "place_in"
    assert result.target == "the top drawer of the cabinet"


def test_pick_then_put_in_relation():
    result = parse_goal_relation("pick up the ketchup and put it in the tray")
    assert result.relation == "place_in"
    assert result.manipulated_object == "the ketchup"
    assert result.target == "the tray"


def test_pick_then_place_directional_relation():
    result = parse_goal_relation("pick up the white mug and place it to the right of the caddy")
    assert result.relation == "place_right_of"
    assert result.manipulated_object == "the white mug"
    assert result.target == "the caddy"


def test_close_is_not_misrepresented_as_containment():
    result = parse_goal_relation("close the top drawer of the cabinet")
    assert result.relation == "close"
    assert result.manipulated_object == "the top drawer"
    assert result.target == "the cabinet"


def test_unknown_language_is_explicitly_unsupported():
    result = parse_goal_relation("rotate the faucet clockwise")
    assert result.relation == "unsupported"
    assert result.target is None


def test_visual_query_removes_article_and_support_relation():
    assert visual_entity_query("the top drawer of the cabinet") == "top drawer"
    assert visual_entity_query("the black bowl") == "black bowl"
    assert visual_category_query("the black bowl") == "bowl"
