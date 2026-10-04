#!/usr/bin/env python3
"""Build a complete, task-agnostic visual relation coverage matrix for LIBERO.

The input is produced by audit_libero_goal_predicates.py.  This script does
not use episode outcomes or benchmark labels; it only maps the public task
specification language and goal predicates to reusable perception adapters.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path


SELECTOR_PATTERNS = {
    "left": r"\b(left|leftmost)\b",
    "right": r"\b(right|rightmost)\b",
    "front": r"\b(front|frontmost)\b",
    "back": r"\b(back|backmost)\b",
    "middle": r"\b(middle|center|central)\b",
    "between": r"\bbetween\b",
    "top": r"\b(top|upper)\b",
    "bottom": r"\b(bottom|lower)\b",
}


def target_subtype(predicate: str, arguments: list[str]) -> str:
    """Classify a goal into a reusable relation adapter, not a task ID."""
    p = predicate.lower()
    if p == "in":
        target = arguments[1].lower()
        if "heating_region" in target:
            return "containment/microwave_cavity"
        if "contain_region" in target:
            return "containment/container_interior"
        if re.search(r"_(top|middle|bottom)_region$", target):
            return "containment/drawer_interior"
        return "containment/generic_region"
    if p == "on":
        target = arguments[1].lower()
        if "cook_region" in target:
            return "support/appliance_surface"
        if "top_side" in target:
            return "support/object_top_surface"
        if target.endswith("_region"):
            return "support/workspace_region"
        return "support/object_or_stack"
    if p in {"open", "close"}:
        target = arguments[0].lower()
        if "microwave" in target:
            return "articulation/microwave_door"
        if re.search(r"_(top|middle|bottom)_region$", target):
            return "articulation/drawer"
        return "articulation/generic"
    if p in {"turnon", "turnoff"}:
        return "device/binary_state"
    return "unsupported"


ADAPTERS = {
    "containment": {
        "measurement": "subject mask overlap with container opening/interior plus boundary crossing and temporal persistence",
        "required_entities": ["subject", "container_or_region"],
        "state_output": ["outside", "crossing_boundary", "inside", "uncertain"],
    },
    "support": {
        "measurement": "subject-support relative geometry, contact-height consistency, relative-motion lock, and post-release persistence",
        "required_entities": ["subject", "support_or_region"],
        "state_output": ["not_supported", "approaching", "supported", "released_and_stable", "uncertain"],
    },
    "articulation": {
        "measurement": "moving-panel/handle displacement normalized between calibrated open and closed endpoints",
        "required_entities": ["articulated_fixture", "handle_or_panel"],
        "state_output": ["open", "closed", "intermediate", "moving", "uncertain"],
    },
    "device": {
        "measurement": "task-specific observable state channel (indicator/geometry) with temporal confirmation; otherwise unknown, never guessed",
        "required_entities": ["device", "state_indicator_if_observable"],
        "state_output": ["on", "off", "uncertain", "unobservable"],
    },
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    audit = json.loads(args.input.read_text(encoding="utf-8"))
    subtype_counts: collections.Counter[str] = collections.Counter()
    selector_counts: collections.Counter[str] = collections.Counter()
    suite_subtypes: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    records = []
    unsupported = []

    for record in audit["records"]:
        language = record["language"].lower()
        selectors = [name for name, pattern in SELECTOR_PATTERNS.items() if re.search(pattern, language)]
        selector_counts.update(selectors)
        goals = []
        for goal in record["goals"]:
            subtype = target_subtype(goal["predicate"], goal["arguments"])
            family = subtype.split("/", 1)[0]
            subtype_counts[subtype] += 1
            suite_subtypes[record["suite"]][subtype] += 1
            enriched = {**goal, "relation_family": family, "adapter_subtype": subtype}
            goals.append(enriched)
            if family not in ADAPTERS:
                unsupported.append({"file": record["file"], "goal": enriched})
        records.append({**record, "entity_selectors": selectors, "goals": goals})

    result = {
        "schema_version": 1,
        "source_audit": str(args.input),
        "scope": {
            "task_count": audit["task_count"],
            "goal_atom_count": audit["goal_atom_count"],
            "raw_predicates": sorted(audit["predicate_counts"]),
            "relation_families": sorted(ADAPTERS),
        },
        "design_rule": "Entity grounding selects arguments; relation adapters verify predicate truth. Unknown evidence abstains.",
        "adapters": ADAPTERS,
        "adapter_subtype_counts": dict(sorted(subtype_counts.items())),
        "suite_adapter_subtype_counts": {
            suite: dict(sorted(counts.items())) for suite, counts in sorted(suite_subtypes.items())
        },
        "entity_selector_counts": dict(sorted(selector_counts.items())),
        "unsupported_goal_count": len(unsupported),
        "unsupported_goals": unsupported,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "tasks": audit["task_count"],
        "goals": audit["goal_atom_count"],
        "families": sorted(ADAPTERS),
        "subtypes": dict(sorted(subtype_counts.items())),
        "selectors": dict(sorted(selector_counts.items())),
        "unsupported": len(unsupported),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
