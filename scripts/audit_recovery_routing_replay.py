"""Replay frozen supervisor decisions through the recovery router.

This is a causal, read-only audit. It consumes only fields that were available
at each historical decision and never reads episode success while routing.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path

from vla_supervisor.events import EventType, MonitorEvent, RecommendedAction, SupervisorDecision
from vla_supervisor.intervention import InterventionMode, InterventionPlanner


def _event(value: dict) -> MonitorEvent:
    return MonitorEvent(
        EventType(value["event_type"]),
        float(value.get("score", 0.0)),
        float(value.get("confidence", 0.0)),
        evidence=value.get("evidence") or {},
        recommended_action=RecommendedAction(value.get("recommended_action", "continue")),
        source=str(value.get("source", "unknown")),
        action_index=value.get("action_index"),
        timestamp=value.get("timestamp"),
    )


def _decision(value: dict, events: tuple[MonitorEvent, ...]) -> SupervisorDecision:
    return SupervisorDecision(
        RecommendedAction(value["action"]),
        EventType(value["reason"]),
        float(value.get("confidence", 0.0)),
        events,
        bool(value.get("clear_remaining_chunk", False)),
        bool(value.get("request_replan", False)),
        int(value.get("cooldown_remaining", 0)),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    planner = InterventionPlanner()
    records = []
    route_counts: Counter[str] = Counter()
    target_counts: Counter[str] = Counter()
    mode_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    unsafe_low_confidence_primitive = 0

    for path in args.inputs:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("event") != "supervisor_decision":
                continue
            decision_value = row.get("decision") or {}
            if not decision_value.get("request_replan", False):
                continue
            events = tuple(_event(value) for value in row.get("events", []))
            decision = _decision(decision_value, events)
            directive = planner.plan(decision, events, last_action=None)
            # With no historical last_action, the replay intentionally cannot
            # invent a retreat vector. Prompt routing remains fully auditable.
            risk_bearing_primitive = directive.mode is InterventionMode.RETREAT_THEN_REPLAN
            if (risk_bearing_primitive
                    and directive.mechanism_confidence < planner.config.mechanism_action_confidence):
                unsafe_low_confidence_primitive += 1
            route_counts[directive.recovery_mechanism] += 1
            target_counts[directive.recovery_target] += 1
            mode_counts[directive.mode.value] += 1
            reason_counts[decision.reason.value] += 1
            records.append({
                "source": str(path),
                "line_number": line_number,
                "action_index": row.get("action_index"),
                "decision_reason": decision.reason.value,
                "route": directive.recovery_mechanism,
                "recovery_target": directive.recovery_target,
                "recovery_target_confidence": directive.recovery_target_confidence,
                "mechanism_confidence": directive.mechanism_confidence,
                "mode_without_action_vector": directive.mode.value,
                "bridge_action_count": len(directive.bridge_actions),
                "recovery_prompt_suffix": directive.recovery_prompt_suffix,
            })

    result = {
        "schema_version": 1,
        "causal_contract": (
            "routing uses historical step-local monitor events only; episode outcome is not read"
        ),
        "input_files": len(args.inputs),
        "replan_decisions": len(records),
        "reason_counts": dict(reason_counts),
        "route_counts": dict(route_counts),
        "recovery_target_counts": dict(target_counts),
        "mode_counts_without_action_vector": dict(mode_counts),
        "unsafe_low_confidence_primitive_count": unsafe_low_confidence_primitive,
        "records": records,
    }
    if unsafe_low_confidence_primitive:
        raise RuntimeError("low-confidence mechanism generated a physical primitive")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "records"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
