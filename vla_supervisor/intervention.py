"""Conservative intervention planning after supervisor decisions.

This module does not execute robot actions.  It converts fused monitor events
into an auditable directive for the online loop.  Every bridge action must
still pass the normal instruction guard before execution.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Iterable, Sequence

from .embodiment import CartesianDeltaEmbodiment, RecoveryEmbodiment
from .events import EventType, MonitorEvent, RecommendedAction, SupervisorDecision
from .recovery_targets import RecoveryTarget, RecoveryTargetRouter


class InterventionMode(str, Enum):
    NOMINAL = "nominal"
    CAUTIOUS_CONTINUE = "cautious_continue"
    REPLAN_SHORT = "replan_short"
    HOLD_THEN_REPLAN = "hold_then_replan"
    RETREAT_THEN_REPLAN = "retreat_then_replan"
    SAFE_STOP = "safe_stop"


@dataclass(frozen=True)
class InterventionConfig:
    nominal_horizon: int = 5
    cautious_horizon: int = 1
    act_on_ambiguity: bool = True
    retreat_on_policy_stall: bool = False
    hold_steps: int = 1
    retreat_steps: int = 2
    retreat_translation_magnitude: float = 0.15
    release_steps: int = 1
    mechanism_action_confidence: float = 0.75

    def __post_init__(self):
        if self.nominal_horizon < 1 or self.cautious_horizon < 1:
            raise ValueError("chunk horizons must be positive")
        if self.cautious_horizon > self.nominal_horizon:
            raise ValueError("cautious horizon cannot exceed nominal horizon")
        if self.hold_steps < 0 or self.retreat_steps < 0 or self.release_steps < 0:
            raise ValueError("bridge step counts must be non-negative")
        if not 0.0 <= self.retreat_translation_magnitude <= 0.25:
            raise ValueError("retreat magnitude must be in [0, 0.25]")
        if not 0.0 <= self.mechanism_action_confidence <= 1.0:
            raise ValueError("mechanism_action_confidence must be in [0, 1]")


@dataclass(frozen=True)
class InterventionDirective:
    mode: InterventionMode
    next_chunk_horizon: int
    truncate_remaining_to: int | None = None
    bridge_actions: tuple[tuple[float, ...], ...] = ()
    replan_after_bridge: bool = False
    explanation: str = ""
    recovery_mechanism: str = "none"
    mechanism_confidence: float = 0.0
    recovery_prompt_suffix: str = ""
    recovery_target: str = RecoveryTarget.REQUEST_MORE_EVIDENCE.value
    recovery_target_confidence: float = 0.0

    def to_dict(self) -> dict:
        value = asdict(self)
        value["mode"] = self.mode.value
        return value


class InterventionPlanner:
    def __init__(self, config: InterventionConfig = InterventionConfig(),
                 embodiment: RecoveryEmbodiment | None = None,
                 recovery_target_router: RecoveryTargetRouter | None = None):
        self.config = config
        self.embodiment = embodiment or CartesianDeltaEmbodiment()
        self.recovery_target_router = recovery_target_router or RecoveryTargetRouter(
            config.mechanism_action_confidence)

    @staticmethod
    def _diagnostic_states(events: Iterable[MonitorEvent]) -> set[str]:
        return {
            str(event.evidence.get("diagnostic_state"))
            for event in events if event.evidence.get("diagnostic_state") is not None
        }

    @staticmethod
    def _mechanism_context(events: Iterable[MonitorEvent], reason: EventType):
        """Return a task-agnostic mechanism, confidence, and prompt fragment.

        The fragment describes what must be re-observed or re-established; it
        never names a LIBERO task, object, or privileged simulator state.
        """
        events = tuple(events)
        priority = (
            ("wrong_object_control", "wrong_object_control",
             "Immediate recovery: the gripper may control the wrong object. "
             "Re-observe the intended target, release only into a safe area, "
             "then approach and grasp the intended object."),
            ("object_loss_risk", "object_loss_risk",
             "Immediate recovery: the previously controlled object may have "
             "been lost. Re-observe its current location and re-establish a "
             "stable grasp before continuing transport."),
            ("empty_grasp_or_miss", "empty_grasp_or_miss",
             "Immediate recovery: the grasp may be empty or misaligned. "
             "Re-open if needed, move clear, re-observe the target, and retry "
             "the grasp from the current scene."),
            ("fixed_obstacle_or_jam", "fixed_obstacle_or_jam",
             "Immediate recovery: motion may be obstructed. Stop pushing in "
             "the current direction, move clear, re-observe, and choose a "
             "collision-free approach."),
        )
        for diagnostic_state, mechanism, prompt in priority:
            matches = [event for event in events
                       if str(event.evidence.get("diagnostic_state")) == diagnostic_state]
            if matches:
                return mechanism, max(float(event.confidence) for event in matches), prompt
        if reason is EventType.POLICY_STALL:
            return ("policy_stall", max((float(x.confidence) for x in events), default=0.0),
                    "Immediate recovery: task progress appears stalled. "
                    "Re-observe the scene, identify the next unmet task "
                    "precondition, and retry that stage from the current state.")
        if reason is EventType.EXECUTION_MISMATCH:
            return ("execution_mismatch", max((float(x.confidence) for x in events), default=0.0),
                    "Immediate recovery: the observed motion did not match the "
                    "planned motion. Re-observe the robot and scene, use a "
                    "short cautious motion, and avoid repeating the same push.")
        if reason is EventType.INSTRUCTION_UNSAFE:
            return ("instruction_unsafe", max((float(x.confidence) for x in events), default=0.0),
                    "Immediate recovery: the previous unexecuted motion was "
                    "rejected by the safety guard. Re-observe and choose a "
                    "smaller safe action toward the original task.")
        return ("unknown", max((float(x.confidence) for x in events), default=0.0),
                "Immediate recovery: execution is off nominal. Re-observe the "
                "current scene, reassess the next task precondition, and use "
                "short cautious actions without assuming the prior plan remains valid.")

    def _hold_action(self, last_action: Sequence[float] | None) -> tuple[float, ...]:
        return self.embodiment.hold_action(last_action)

    def _retreat_action(self, last_action: Sequence[float] | None) -> tuple[float, ...] | None:
        return self.embodiment.retreat_action(
            last_action, self.config.retreat_translation_magnitude)

    def _release_action(self, reference_action: Sequence[float] | None, *,
                        preserve_motion: bool = False) -> tuple[float, ...] | None:
        release = getattr(self.embodiment, "release_action", None)
        return None if release is None else release(
            reference_action, preserve_motion=preserve_motion)

    def plan(self, decision: SupervisorDecision, events: Iterable[MonitorEvent],
             last_action: Sequence[float] | None = None) -> InterventionDirective:
        events = tuple(events)
        if decision.action is RecommendedAction.SAFE_STOP:
            return InterventionDirective(
                InterventionMode.SAFE_STOP, self.config.cautious_horizon,
                explanation="recovery budget exhausted or explicit safe stop",
            )

        diagnostic_states = self._diagnostic_states(events)
        if decision.request_replan:
            mechanism, mechanism_confidence, recovery_prompt = self._mechanism_context(
                events, decision.reason)
            target_decision = self.recovery_target_router.route(events)
            if target_decision.actionable:
                recovery_prompt = "Immediate recovery milestone: " + target_decision.prompt_suffix
            mechanism_actionable = (
                mechanism_confidence >= self.config.mechanism_action_confidence)
            release_then_retreat_states = {
                "empty_grasp_or_miss", "wrong_object_control", "object_loss_risk",
            }
            if mechanism_actionable and diagnostic_states & release_then_retreat_states:
                release = self._release_action(last_action)
                retreat = self._retreat_action(last_action)
                bridge = []
                if release is not None:
                    bridge.extend([release] * self.config.release_steps)
                if retreat is not None:
                    open_retreat = self._release_action(retreat, preserve_motion=True)
                    bridge.extend([open_retreat or retreat] * self.config.retreat_steps)
                if bridge:
                    return InterventionDirective(
                        InterventionMode.RETREAT_THEN_REPLAN,
                        self.config.cautious_horizon,
                        bridge_actions=tuple(bridge),
                        replan_after_bridge=True,
                        explanation=("release and bounded retreat after independently "
                                     "diagnosed grasp/control failure"),
                        recovery_mechanism=mechanism,
                        mechanism_confidence=mechanism_confidence,
                        recovery_prompt_suffix=recovery_prompt,
                        recovery_target=target_decision.target.value,
                        recovery_target_confidence=target_decision.confidence,
                    )
            should_retreat = (
                (mechanism_actionable and "fixed_obstacle_or_jam" in diagnostic_states)
                or (self.config.retreat_on_policy_stall
                    and decision.reason is EventType.POLICY_STALL)
            )
            if should_retreat:
                retreat = self._retreat_action(last_action)
                if retreat is not None:
                    return InterventionDirective(
                        InterventionMode.RETREAT_THEN_REPLAN,
                        self.config.cautious_horizon,
                        bridge_actions=(retreat,) * self.config.retreat_steps,
                        replan_after_bridge=True,
                        explanation=("bounded reverse motion after persistent stall"
                                     if decision.reason is EventType.POLICY_STALL else
                                     "bounded reverse motion after independently diagnosed jam"),
                        recovery_mechanism=mechanism,
                        mechanism_confidence=mechanism_confidence,
                        recovery_prompt_suffix=recovery_prompt,
                        recovery_target=target_decision.target.value,
                        recovery_target_confidence=target_decision.confidence,
                    )
            if decision.reason is EventType.INSTRUCTION_UNSAFE:
                return InterventionDirective(
                    InterventionMode.REPLAN_SHORT, self.config.cautious_horizon,
                    replan_after_bridge=True,
                    explanation="unsafe unexecuted instruction; discard and replan",
                    recovery_mechanism=mechanism,
                    mechanism_confidence=mechanism_confidence,
                    recovery_prompt_suffix=recovery_prompt,
                    recovery_target=target_decision.target.value,
                    recovery_target_confidence=target_decision.confidence,
                )
            hold = self._hold_action(last_action)
            return InterventionDirective(
                InterventionMode.HOLD_THEN_REPLAN,
                self.config.cautious_horizon,
                bridge_actions=(hold,) * self.config.hold_steps,
                replan_after_bridge=True,
                explanation=("freeze translation/rotation before replanning; "
                             "mechanism-specific motion withheld when confidence is insufficient"),
                recovery_mechanism=mechanism,
                mechanism_confidence=mechanism_confidence,
                recovery_prompt_suffix=recovery_prompt,
                recovery_target=target_decision.target.value,
                recovery_target_confidence=target_decision.confidence,
            )

        ambiguous = any(event.event_type in {
            EventType.EXECUTION_AMBIGUOUS, EventType.OBJECT_AMBIGUOUS
        } for event in events)
        if ambiguous:
            if not self.config.act_on_ambiguity:
                return InterventionDirective(
                    InterventionMode.CAUTIOUS_CONTINUE,
                    self.config.nominal_horizon,
                    explanation=("record ambiguity without queue mutation until an "
                                 "independent diagnostic modality is online"),
                )
            return InterventionDirective(
                InterventionMode.CAUTIOUS_CONTINUE,
                self.config.cautious_horizon,
                truncate_remaining_to=self.config.cautious_horizon,
                explanation="retain feedback control while collecting ambiguity evidence",
            )
        return InterventionDirective(
            InterventionMode.NOMINAL, self.config.nominal_horizon,
            explanation="no actionable or ambiguous evidence",
        )
