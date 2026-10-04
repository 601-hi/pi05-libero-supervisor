from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Mapping


class EventType(str, Enum):
    NORMAL = "normal"
    INSTRUCTION_UNSAFE = "instruction_unsafe"
    EXECUTION_MISMATCH = "execution_mismatch"
    EXECUTION_AMBIGUOUS = "execution_ambiguous"
    POLICY_STALL = "policy_stall"
    OBJECT_FAILURE = "object_failure"
    OBJECT_AMBIGUOUS = "object_ambiguous"


class RecommendedAction(str, Enum):
    CONTINUE = "continue"
    SLOW_OR_CLIP = "slow_or_clip"
    STOP_CHUNK = "stop_chunk"
    STOP_AND_REPLAN = "stop_and_replan"
    REQUEST_MORE_EVIDENCE = "request_more_evidence"
    SAFE_STOP = "safe_stop"


@dataclass(frozen=True)
class MonitorEvent:
    event_type: EventType
    score: float
    confidence: float
    evidence: Mapping[str, Any] = field(default_factory=dict)
    recommended_action: RecommendedAction = RecommendedAction.CONTINUE
    source: str = "unknown"
    action_index: int | None = None
    timestamp: float | None = None

    def __post_init__(self):
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["event_type"] = self.event_type.value
        value["recommended_action"] = self.recommended_action.value
        return value


@dataclass(frozen=True)
class SupervisorDecision:
    action: RecommendedAction
    reason: EventType
    confidence: float
    triggering_events: tuple[MonitorEvent, ...] = ()
    clear_remaining_chunk: bool = False
    request_replan: bool = False
    cooldown_remaining: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action.value,
            "reason": self.reason.value,
            "confidence": self.confidence,
            "triggering_events": [x.to_dict() for x in self.triggering_events],
            "clear_remaining_chunk": self.clear_remaining_chunk,
            "request_replan": self.request_replan,
            "cooldown_remaining": self.cooldown_remaining,
        }
