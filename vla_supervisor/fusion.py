from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass

from .events import EventType, MonitorEvent, RecommendedAction, SupervisorDecision


@dataclass(frozen=True)
class PersistenceRule:
    required: int = 2
    window: int = 3
    minimum_confidence: float = 0.5


class TemporalFusion:
    def __init__(self, rules: dict[EventType, PersistenceRule] | None = None):
        default = PersistenceRule()
        # Ambiguous evidence is logged and routed to another modality; it is
        # deliberately non-actionable and must never acquire a default rule.
        actionable = (EventType.INSTRUCTION_UNSAFE, EventType.EXECUTION_MISMATCH,
                      EventType.POLICY_STALL, EventType.OBJECT_FAILURE)
        self.rules = {kind: default for kind in actionable}
        self.rules[EventType.INSTRUCTION_UNSAFE] = PersistenceRule(1, 1, 1.0)
        if rules: self.rules.update(rules)
        self.reset()

    def reset(self):
        self.history = defaultdict(deque)

    def update(self, events: list[MonitorEvent]) -> SupervisorDecision:
        current = {event.event_type: event for event in events if event.event_type is not EventType.NORMAL}
        triggered = []
        for kind, rule in self.rules.items():
            queue = self.history[kind]; queue.append(current.get(kind))
            while len(queue) > rule.window: queue.popleft()
            qualifying = [x for x in queue if x is not None and x.confidence >= rule.minimum_confidence]
            if len(queue) == rule.window and len(qualifying) >= rule.required:
                triggered.extend(qualifying)
        if not triggered:
            return SupervisorDecision(RecommendedAction.CONTINUE, EventType.NORMAL, 1.0)
        priority = {EventType.INSTRUCTION_UNSAFE: 4, EventType.EXECUTION_MISMATCH: 3,
                    EventType.OBJECT_FAILURE: 2, EventType.POLICY_STALL: 1}
        reason = max((x.event_type for x in triggered), key=priority.get)
        selected = tuple(x for x in triggered if x.event_type is reason)
        return SupervisorDecision(RecommendedAction.STOP_AND_REPLAN, reason,
                                  max(x.confidence for x in selected), selected,
                                  clear_remaining_chunk=True, request_replan=True)
