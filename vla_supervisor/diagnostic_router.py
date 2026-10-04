"""Selective routing from execution ambiguity to independent visual diagnosis."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .events import EventType, MonitorEvent, RecommendedAction


@dataclass(frozen=True)
class VisualRoutingConfig:
    """Conservative trigger policy for the expensive visual consequence layer."""

    trigger_after: int = 1
    keep_alive_steps: int = 2
    minimum_execution_confidence: float = 0.0

    def __post_init__(self):
        if self.trigger_after < 1 or self.keep_alive_steps < 0:
            raise ValueError("invalid visual routing window")
        if not 0.0 <= self.minimum_execution_confidence <= 1.0:
            raise ValueError("minimum_execution_confidence must be in [0, 1]")


class AmbiguityGatedObjectMonitor:
    """Call object vision only after execution evidence becomes ambiguous.

    Clear execution mismatch remains the responsibility of the execution layer;
    visual evidence is not allowed to veto it.  Normal execution does not pay
    the visual inference cost.  The wrapped visual monitor receives the same
    information-firewalled history it received before this router was added.
    """

    name = "ambiguity_gated_object_diagnosis"

    def __init__(self, object_monitor, config: VisualRoutingConfig = VisualRoutingConfig(), phase_gate=None):
        self.object_monitor = object_monitor
        self.config = config
        self.phase_gate = phase_gate
        self.reset()

    def reset(self):
        self.ambiguous_run = 0
        self.keep_alive_remaining = 0
        self.invocations = 0
        self.object_monitor.reset()
        if self.phase_gate is not None:
            self.phase_gate.reset()

    def _route(self, upstream_events: Iterable[MonitorEvent]) -> tuple[bool, str]:
        events = tuple(upstream_events)
        actionable = any(
            event.event_type in {EventType.INSTRUCTION_UNSAFE, EventType.EXECUTION_MISMATCH}
            for event in events
        )
        if actionable:
            self.ambiguous_run = 0
            self.keep_alive_remaining = 0
            return False, "clear_upstream_decision"

        ambiguous = any(
            event.event_type is EventType.EXECUTION_AMBIGUOUS
            and event.confidence >= self.config.minimum_execution_confidence
            for event in events
        )
        self.ambiguous_run = self.ambiguous_run + 1 if ambiguous else 0
        if self.ambiguous_run >= self.config.trigger_after:
            self.keep_alive_remaining = self.config.keep_alive_steps
            return True, "execution_ambiguous"
        if self.keep_alive_remaining > 0:
            self.keep_alive_remaining -= 1
            return True, "ambiguity_followup"
        return False, "execution_not_ambiguous"

    def observe_conditioned(self, *, upstream_events, images_before, images_after,
                            intended_action, history, action_index):
        if self.phase_gate is not None and not self.phase_gate.update(intended_action):
            self.ambiguous_run = 0
            self.keep_alive_remaining = 0
            return MonitorEvent(
                EventType.NORMAL, 0.0, 1.0,
                {"visual_invoked": False, "routing_reason": "outside_manipulation_phase"},
                RecommendedAction.CONTINUE, self.name, action_index,
            )
        invoke, reason = self._route(upstream_events)
        if not invoke:
            return MonitorEvent(
                EventType.NORMAL,
                0.0,
                1.0,
                {"visual_invoked": False, "routing_reason": reason},
                RecommendedAction.CONTINUE,
                self.name,
                action_index,
            )
        self.invocations += 1
        contextual_observe = getattr(self.object_monitor, "observe_with_execution_context", None)
        object_kwargs = {
            "images_before": images_before,
            "images_after": images_after,
            "intended_action": intended_action,
            "history": history,
            "action_index": action_index,
        }
        if contextual_observe is not None:
            result = contextual_observe(upstream_events=upstream_events, **object_kwargs)
        else:
            result = self.object_monitor.observe(**object_kwargs)
        return MonitorEvent(
            result.event_type,
            result.score,
            result.confidence,
            {**dict(result.evidence), "visual_invoked": True, "routing_reason": reason},
            result.recommended_action,
            result.source,
            result.action_index,
            result.timestamp,
        )

    def observe(self, **kwargs):
        """Fail closed against accidental unconditional use outside the runtime."""
        raise RuntimeError("AmbiguityGatedObjectMonitor requires upstream execution events")


class ShadowObjectMonitor:
    """Run a real object diagnostic while preventing it from changing control.

    Actionable diagnoses remain fully observable in evidence, but are emitted
    as ambiguity until a frozen paired evaluation authorizes online control.
    """

    name = "shadow_object_diagnosis"

    def __init__(self, object_monitor):
        self.object_monitor = object_monitor

    def reset(self):
        self.object_monitor.reset()

    @staticmethod
    def _shadow(result: MonitorEvent) -> MonitorEvent:
        if result.event_type is not EventType.OBJECT_FAILURE:
            return result
        return MonitorEvent(
            EventType.OBJECT_AMBIGUOUS,
            result.score,
            result.confidence,
            {
                **dict(result.evidence),
                "shadow_original_event_type": result.event_type.value,
                "shadow_original_recommended_action": result.recommended_action.value,
                "shadow_control_enabled": False,
            },
            RecommendedAction.REQUEST_MORE_EVIDENCE,
            result.source,
            result.action_index,
            result.timestamp,
        )

    def observe_with_execution_context(self, **kwargs):
        observe = getattr(self.object_monitor, "observe_with_execution_context", None)
        if observe is None:
            result = self.object_monitor.observe(**kwargs)
        else:
            result = observe(**kwargs)
        return self._shadow(result)

    def observe(self, **kwargs):
        return self._shadow(self.object_monitor.observe(**kwargs))


@dataclass
class GripperInteractionGate:
    """Causal manipulation-phase gate from gripper command transitions.

    A stable sign reversal with sufficient magnitude marks a close/release
    event.  Vision is enabled for a bounded causal window after that event.
    The first command only initializes state and cannot trigger the gate.
    """

    active_steps_after_transition: int = 40
    command_threshold: float = 0.5

    def __post_init__(self):
        if self.active_steps_after_transition < 1 or not 0 <= self.command_threshold <= 1:
            raise ValueError("invalid gripper interaction gate")
        self.reset()

    def reset(self):
        self.previous_sign = None
        self.remaining = 0

    def update(self, intended_action) -> bool:
        values = list(intended_action)
        command = float(values[6]) if len(values) > 6 else 0.0
        sign = 1 if command >= self.command_threshold else -1 if command <= -self.command_threshold else 0
        transition = self.previous_sign in {-1, 1} and sign in {-1, 1} and sign != self.previous_sign
        if sign in {-1, 1}:
            self.previous_sign = sign
        if transition:
            self.remaining = self.active_steps_after_transition
        active = self.remaining > 0
        if self.remaining > 0:
            self.remaining -= 1
        return active
