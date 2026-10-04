"""Causal contact, grasp, semantic identity, and attachment diagnosis.

This module consumes calibrated evidence.  It deliberately does not perform
image recognition itself, so background-flow, wrist tracking, and semantic
models remain independently replaceable and auditable.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Mapping, Optional

from .events import EventType, MonitorEvent, RecommendedAction


class ContactGraspState(str, Enum):
    FREE_MOTION = "free_motion"
    CONTACT_SUSPECTED = "contact_suspected"
    FIXED_OBSTACLE_OR_JAM = "fixed_obstacle_or_jam"
    EMPTY_GRASP_OR_MISS = "empty_grasp_or_miss"
    OBJECT_CONTROL_ESTABLISHED = "object_control_established"
    WRONG_OBJECT_CONTROL = "wrong_object_control"
    TARGET_CONTROLLED = "target_controlled"
    HEALTHY_TRANSPORT = "healthy_transport"
    TRANSIENT_UNCERTAINTY = "transient_uncertainty"
    OBJECT_LOSS_RISK = "object_loss_risk"


@dataclass(frozen=True)
class ContactEvidence:
    """One causal time step of already calibrated multimodal evidence.

    ``execution_decision`` follows the dual-expert four-state vocabulary.
    Probabilities should be omitted (``None``) when the corresponding sensor
    cannot support a conclusion; missing evidence must never become negative
    evidence implicitly.
    """

    execution_decision: str
    execution_reliability: float
    manipulation_phase: bool
    gripper_closed: bool
    background_confidence: float
    controlled_object_candidate_probability: Optional[float] = None
    gripper_candidate_proximity: Optional[float] = None
    independent_motion_probability: Optional[float] = None
    attachment_probability: Optional[float] = None
    blocked_motion_probability: Optional[float] = None
    semantic_target_probability: Optional[float] = None


@dataclass(frozen=True)
class ContactDiagnosisConfig:
    evidence_threshold: float = 0.7
    background_confidence_threshold: float = 0.5
    contact_confirm_steps: int = 2
    consequence_observation_steps: int = 8
    control_confirm_steps: int = 3
    loss_confirm_steps: int = 3
    recovery_confirm_steps: int = 3
    semantic_threshold: float = 0.7

    def __post_init__(self):
        probabilities = (
            self.evidence_threshold,
            self.background_confidence_threshold,
            self.semantic_threshold,
        )
        if any(not 0.0 <= value <= 1.0 for value in probabilities):
            raise ValueError("probability thresholds must be in [0, 1]")
        counts = (
            self.contact_confirm_steps,
            self.consequence_observation_steps,
            self.control_confirm_steps,
            self.loss_confirm_steps,
            self.recovery_confirm_steps,
        )
        if any(value < 1 for value in counts):
            raise ValueError("temporal thresholds must be positive")


@dataclass(frozen=True)
class ContactDiagnosis:
    state: ContactGraspState
    confidence: float
    actionable: bool
    evidence: Mapping[str, Any]


class ContactGraspSemanticStateMachine:
    """Causal diagnostic state machine with explicit abstention.

    Low dual-expert reliability is only a routing condition.  Contact requires
    independent proximity/background support, and object control requires
    future independent-motion plus wrist-attachment evidence.
    """

    _UNCERTAIN_EXECUTION = {"ambiguous_overlap", "unknown"}

    def __init__(self, config: ContactDiagnosisConfig = ContactDiagnosisConfig()):
        self.config = config
        self.reset()

    def reset(self):
        self.state = ContactGraspState.FREE_MOTION
        self.contact_run = 0
        self.observation_age = 0
        self.control_run = 0
        self.blocked_run = 0
        self.no_object_run = 0
        self.loss_run = 0
        self.recovery_run = 0
        self.control_ever_established = False

    @staticmethod
    def _present_high(value: Optional[float], threshold: float) -> bool:
        return value is not None and value >= threshold

    def _diagnosis(self, evidence: ContactEvidence, reason: str, *, actionable=False,
                   confidence=0.0) -> ContactDiagnosis:
        return ContactDiagnosis(
            self.state,
            float(max(0.0, min(1.0, confidence))),
            bool(actionable),
            {
                "reason": reason,
                "contact_run": self.contact_run,
                "observation_age": self.observation_age,
                "control_run": self.control_run,
                "blocked_run": self.blocked_run,
                "no_object_run": self.no_object_run,
                "loss_run": self.loss_run,
                "recovery_run": self.recovery_run,
                "execution_decision": evidence.execution_decision,
                "execution_reliability": evidence.execution_reliability,
                "controlled_object_candidate_probability": (
                    evidence.controlled_object_candidate_probability
                ),
            },
        )

    def update(self, evidence: ContactEvidence) -> ContactDiagnosis:
        cfg = self.config
        if evidence.execution_decision not in {
            "known_normal", "known_abnormal", "ambiguous_overlap", "unknown"
        }:
            raise ValueError("invalid execution_decision")
        if not 0.0 <= evidence.execution_reliability <= 1.0:
            raise ValueError("execution_reliability must be in [0, 1]")

        background_valid = evidence.background_confidence >= cfg.background_confidence_threshold
        close_support = self._present_high(evidence.gripper_candidate_proximity, cfg.evidence_threshold)
        routing_uncertain = evidence.execution_decision in self._UNCERTAIN_EXECUTION
        contact_support = (
            evidence.manipulation_phase
            and evidence.gripper_closed
            and background_valid
            and close_support
            and routing_uncertain
        )

        if self.state is ContactGraspState.FREE_MOTION:
            self.contact_run = self.contact_run + 1 if contact_support else 0
            if self.contact_run >= cfg.contact_confirm_steps:
                self.state = ContactGraspState.CONTACT_SUSPECTED
                self.observation_age = 0
                return self._diagnosis(evidence, "independently_supported_contact_hypothesis",
                                       confidence=min(evidence.background_confidence,
                                                      evidence.gripper_candidate_proximity or 0.0))
            return self._diagnosis(evidence, "no_confirmed_contact")

        if self.state is ContactGraspState.CONTACT_SUSPECTED:
            self.observation_age += 1
            moving = self._present_high(evidence.independent_motion_probability, cfg.evidence_threshold)
            attached = self._present_high(evidence.attachment_probability, cfg.evidence_threshold)
            blocked = self._present_high(evidence.blocked_motion_probability, cfg.evidence_threshold)
            self.control_run = self.control_run + 1 if moving and attached else 0
            self.blocked_run = self.blocked_run + 1 if blocked and not moving else 0
            self.no_object_run = self.no_object_run + 1 if not moving and not blocked else 0

            if self.control_run >= cfg.control_confirm_steps:
                self.state = ContactGraspState.OBJECT_CONTROL_ESTABLISHED
                self.control_ever_established = True
                return self._diagnosis(evidence, "independent_motion_and_wrist_attachment_confirmed",
                                       confidence=min(evidence.independent_motion_probability or 0.0,
                                                      evidence.attachment_probability or 0.0))
            if self.blocked_run >= cfg.control_confirm_steps:
                self.state = ContactGraspState.FIXED_OBSTACLE_OR_JAM
                return self._diagnosis(evidence, "blocked_without_independent_object_motion",
                                       actionable=True,
                                       confidence=evidence.blocked_motion_probability or 0.0)
            if self.observation_age >= cfg.consequence_observation_steps:
                self.state = ContactGraspState.EMPTY_GRASP_OR_MISS
                return self._diagnosis(evidence, "no_controlled_object_in_observation_window",
                                       actionable=True,
                                       confidence=min(1.0, self.no_object_run / self.observation_age))
            return self._diagnosis(evidence, "waiting_for_contact_consequence")

        if self.state is ContactGraspState.OBJECT_CONTROL_ESTABLISHED:
            semantic = evidence.semantic_target_probability
            if semantic is None:
                return self._diagnosis(evidence, "semantic_identity_unavailable")
            if semantic >= cfg.semantic_threshold:
                self.state = ContactGraspState.TARGET_CONTROLLED
                return self._diagnosis(evidence, "controlled_object_matches_language_target",
                                       confidence=semantic)
            if semantic <= 1.0 - cfg.semantic_threshold:
                self.state = ContactGraspState.WRONG_OBJECT_CONTROL
                return self._diagnosis(evidence, "controlled_object_conflicts_with_language_target",
                                       actionable=True, confidence=1.0-semantic)
            return self._diagnosis(evidence, "semantic_identity_ambiguous", confidence=0.0)

        if self.state in {ContactGraspState.TARGET_CONTROLLED,
                          ContactGraspState.HEALTHY_TRANSPORT,
                          ContactGraspState.TRANSIENT_UNCERTAINTY}:
            attached = self._present_high(evidence.attachment_probability, cfg.evidence_threshold)
            if attached:
                self.recovery_run += 1
                self.loss_run = 0
                if self.recovery_run >= cfg.recovery_confirm_steps:
                    self.state = ContactGraspState.HEALTHY_TRANSPORT
                    return self._diagnosis(evidence, "attachment_stable_or_recovered",
                                           confidence=evidence.attachment_probability or 0.0)
                return self._diagnosis(evidence, "attachment_recovery_pending")
            self.loss_run += 1
            self.recovery_run = 0
            if self.loss_run >= cfg.loss_confirm_steps:
                self.state = ContactGraspState.OBJECT_LOSS_RISK
                return self._diagnosis(evidence, "persistent_attachment_loss_while_closed",
                                       actionable=True,
                                       confidence=1.0-(evidence.attachment_probability or 0.0))
            self.state = ContactGraspState.TRANSIENT_UNCERTAINTY
            return self._diagnosis(evidence, "transient_attachment_uncertainty")

        return self._diagnosis(evidence, "terminal_diagnostic_state", actionable=True,
                               confidence=1.0)


class ContactDiagnosticObjectMonitor:
    """Adapter from independent visual evidence to supervisor events.

    The scorer sees images, action, and an information-firewalled history.  It
    returns all :class:`ContactEvidence` fields except the execution decision
    and reliability, which are copied from the upstream four-state event.
    """

    name = "contact_grasp_semantic_diagnosis"

    def __init__(self, evidence_scorer: Callable[..., Mapping[str, Any]],
                 config: ContactDiagnosisConfig = ContactDiagnosisConfig()):
        self.evidence_scorer = evidence_scorer
        self.machine = ContactGraspSemanticStateMachine(config)

    def reset(self):
        self.machine.reset()
        reset = getattr(self.evidence_scorer, "reset", None)
        if reset is not None:
            reset()

    @staticmethod
    def _execution_summary(upstream_events) -> tuple[str, float]:
        for event in upstream_events:
            decision = event.evidence.get("decision")
            if decision is not None:
                return str(decision), float(event.evidence.get("reliability", event.confidence))
        return "unknown", 0.0

    def observe_with_execution_context(self, *, upstream_events, images_before, images_after,
                                       intended_action, history, action_index):
        safe_history = tuple({
            "action_index": item.get("action_index"),
            "intended_action": item.get("intended_action"),
        } for item in history)
        visual = dict(self.evidence_scorer(
            images_before, images_after, intended_action, safe_history
        ))
        visual_diagnostics = dict(
            getattr(self.evidence_scorer, "last_diagnostics", {})
        )
        execution_decision, execution_reliability = self._execution_summary(upstream_events)
        evidence = ContactEvidence(
            execution_decision=execution_decision,
            execution_reliability=execution_reliability,
            **visual,
        )
        result = self.machine.update(evidence)
        if result.actionable:
            event_type = EventType.OBJECT_FAILURE
            recommendation = RecommendedAction.STOP_AND_REPLAN
            score = result.confidence
        elif result.state in {
            ContactGraspState.CONTACT_SUSPECTED,
            ContactGraspState.OBJECT_CONTROL_ESTABLISHED,
            ContactGraspState.TRANSIENT_UNCERTAINTY,
        }:
            event_type = EventType.OBJECT_AMBIGUOUS
            recommendation = RecommendedAction.REQUEST_MORE_EVIDENCE
            score = 1.0 - result.confidence
        else:
            event_type = EventType.NORMAL
            recommendation = RecommendedAction.CONTINUE
            score = 0.0
        return MonitorEvent(
            event_type,
            float(score),
            float(result.confidence),
            {**dict(result.evidence), "diagnostic_state": result.state.value,
             "visual_diagnostics": visual_diagnostics},
            recommendation,
            self.name,
            action_index,
        )

    def observe(self, **kwargs):
        raise RuntimeError("ContactDiagnosticObjectMonitor requires upstream execution context")
