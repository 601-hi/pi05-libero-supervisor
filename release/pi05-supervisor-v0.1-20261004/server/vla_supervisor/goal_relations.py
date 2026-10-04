"""Task-id-free parsing of simple manipulation goal relations."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class GoalRelation:
    relation: str
    manipulated_object: str
    target: str | None
    source_relation: str | None = None
    source_description: str | None = None


class TaskGoalAdapter(Protocol):
    """Convert a task specification into carrier-independent relations.

    Implementations may use a planner, an instruction parser, or a structured
    task API. The supervisor core must not branch on benchmark task IDs.
    """
    adapter_id: str
    def parse(self, task_specification: str) -> GoalRelation: ...


def visual_entity_query(entity: str) -> str:
    """Reduce a referring noun phrase to a detector query without task ids."""
    query = re.sub(r"^(?:the|a|an)\s+", "", entity.strip().lower())
    # The manipulated region is the head entity, e.g. "top drawer" rather
    # than the complete support relation "top drawer of the cabinet".
    query = re.split(r"\s+of\s+(?:the\s+)?", query, maxsplit=1)[0]
    return query


def visual_category_query(entity: str) -> str:
    """Remove color adjectives when appearance and language color can diverge."""
    query = visual_entity_query(entity)
    return re.sub(r"^(?:black|white|red|green|blue|yellow|brown|gray|grey)\s+", "", query)


def source_entity_queries(relation: GoalRelation) -> tuple[str, ...]:
    if not relation.source_description:
        return ()
    parts = (
        re.split(r"\s+and\s+", relation.source_description)
        if relation.source_relation == "between"
        else [relation.source_description]
    )
    return tuple(visual_entity_query(part) for part in parts)


_PICK_PLACE = re.compile(
    r"^pick up (?P<object>.+?) (?P<source_relation>between|next to|from|on|in) "
    r"(?P<source>.+?) and place it "
    r"(?P<preposition>on top of|on|in) (?P<target>.+)$"
)
_PICK_PLACE_SIMPLE = re.compile(r"^pick up (?P<object>.+?) and place it (?P<preposition>on top of|on|in) (?P<target>.+)$")
_PICK_PUT_SIMPLE = re.compile(r"^pick up (?P<object>.+?) and put it (?P<preposition>on top of|on|in) (?P<target>.+)$")
_PICK_PLACE_DIRECTIONAL = re.compile(
    r"^pick up (?P<object>.+?) and place it (?P<direction>to the right of|to the left of|"
    r"in front of|behind) (?P<target>.+)$"
)
_PUT = re.compile(r"^put (?P<object>.+?) (?P<preposition>on top of|on|in) (?P<target>.+)$")
_CLOSE = re.compile(r"^close (?P<object>.+?)(?: of (?P<target>.+))?$")
_OPEN = re.compile(r"^open (?P<object>.+?)(?: of (?P<target>.+))?$")
_TURN = re.compile(r"^turn (?P<state>on|off) (?P<object>.+)$")


class SimpleEnglishManipulationGoalAdapter:
    """Small demonstration adapter, not a universal task-language parser."""
    adapter_id = 'simple-english-manipulation-v1'

    def parse(self, task_specification: str) -> GoalRelation:
        text = task_specification.strip().lower().rstrip(".")
        match = _PICK_PLACE.match(text)
        if match:
            return GoalRelation(
                relation='place_in' if match.group('preposition') == 'in' else 'place_on',
                manipulated_object=match.group("object"), target=match.group("target"),
                source_relation=match.group("source_relation").replace(" ", "_"),
                source_description=match.group("source"))
        match = _PICK_PLACE_SIMPLE.match(text) or _PICK_PUT_SIMPLE.match(text) or _PUT.match(text)
        if match:
            return GoalRelation(
                relation='place_in' if match.group('preposition') == 'in' else 'place_on',
                manipulated_object=match.group("object"), target=match.group("target"))
        match = _PICK_PLACE_DIRECTIONAL.match(text)
        if match:
            relation_name = {
                "to the right of": "place_right_of",
                "to the left of": "place_left_of",
                "in front of": "place_in_front_of",
                "behind": "place_behind",
            }[match.group("direction")]
            return GoalRelation(
                relation=relation_name,
                manipulated_object=match.group("object"),
                target=match.group("target"),
            )
        match = _CLOSE.match(text)
        if match: return GoalRelation("close", match.group("object"), match.group("target"))
        match = _OPEN.match(text)
        if match: return GoalRelation("open", match.group("object"), match.group("target"))
        match = _TURN.match(text)
        if match: return GoalRelation(f'turn_{match.group("state")}', match.group("object"), None)
        return GoalRelation("unsupported", text, None)


_LEGACY_ADAPTER = SimpleEnglishManipulationGoalAdapter()


def parse_goal_relation(goal_language: str) -> GoalRelation:
    """Backward-compatible entry point for the demonstration adapter."""
    return _LEGACY_ADAPTER.parse(goal_language)
