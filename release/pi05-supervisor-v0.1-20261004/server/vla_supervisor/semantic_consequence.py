"""Conservative decisions from structured task-consequence evidence.

The upstream vision/VLM component is not trusted to issue robot commands.  It
must provide calibrated, bounded evidence; this module turns that evidence
into a typed diagnosis or explicitly abstains.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .events import EventType, MonitorEvent, RecommendedAction


@dataclass(frozen=True)
class SemanticConsequenceEvidence:
    observable: bool
    calibrated: bool
    goal_satisfied_score: float | None = None
    progress_score: float | None = None
    no_progress_score: float | None = None
    wrong_object_score: float | None = None
    object_lost_score: float | None = None
    metadata: Mapping[str, object] | None = None


@dataclass(frozen=True)
class SemanticConsequenceDecision:
    state: str
    score: float
    confidence: float
    reason: str
    metadata: Mapping[str, object] | None = None


class SemanticConsequenceGate:
    """Require a strong winning score and abstain on conflicts.

    Scores are calibrated evidence values, not assumed posterior probabilities.
    A high anomaly score cannot override equally strong goal-satisfied evidence.
    """

    def __init__(self, *, minimum_score: float = 0.7, minimum_margin: float = 0.15):
        if not 0 <= minimum_score <= 1 or not 0 <= minimum_margin <= 1:
            raise ValueError("semantic consequence thresholds must lie in [0, 1]")
        self.minimum_score = minimum_score
        self.minimum_margin = minimum_margin

    @staticmethod
    def _valid(value: float | None) -> bool:
        return value is not None and 0 <= value <= 1

    def decide(self, evidence: SemanticConsequenceEvidence) -> SemanticConsequenceDecision:
        if not evidence.observable:
            return SemanticConsequenceDecision("unknown", 0.0, 0.0, "task consequence is not visible", evidence.metadata)
        if not evidence.calibrated:
            return SemanticConsequenceDecision("unknown", 0.0, 0.0, "semantic evidence is uncalibrated", evidence.metadata)
        candidates = {
            "goal_achieved": evidence.goal_satisfied_score,
            "progress": evidence.progress_score,
            "no_progress": evidence.no_progress_score,
            "wrong_object": evidence.wrong_object_score,
            "object_lost": evidence.object_lost_score,
        }
        ranked = sorted(
            ((state, float(score)) for state, score in candidates.items() if self._valid(score)),
            key=lambda item: item[1], reverse=True,
        )
        if not ranked:
            return SemanticConsequenceDecision("unknown", 0.0, 0.0, "no valid semantic scores", evidence.metadata)
        state, score = ranked[0]
        runner_up = ranked[1][1] if len(ranked) > 1 else 0.0
        margin = score - runner_up
        if score < self.minimum_score:
            return SemanticConsequenceDecision("ambiguous", score, margin, "best score is below gate", evidence.metadata)
        if margin < self.minimum_margin:
            return SemanticConsequenceDecision("ambiguous", score, margin, "semantic hypotheses conflict", evidence.metadata)
        return SemanticConsequenceDecision(state, score, margin, "strong calibrated semantic evidence", evidence.metadata)


@dataclass(frozen=True)
class SemanticTemporalConfig:
    no_progress_confirmations: int = 2
    immediate_states: tuple[str, ...] = ("wrong_object", "object_lost")

    def __post_init__(self):
        if self.no_progress_confirmations < 1:
            raise ValueError("no_progress_confirmations must be positive")


class SemanticConsequenceMonitor:
    """Convert typed semantic diagnoses to conservative supervisor events."""

    name = "semantic_task_consequence"

    def __init__(self, config: SemanticTemporalConfig = SemanticTemporalConfig()):
        self.config = config
        self.reset()

    def reset(self) -> None:
        self.no_progress_run = 0

    def observe_decision(self, decision: SemanticConsequenceDecision, *, action_index: int) -> MonitorEvent:
        metadata = dict(decision.metadata or {})
        evidence = {"semantic_state": decision.state, "semantic_reason": decision.reason, **metadata}
        if decision.state == "wrong_object":
            evidence["diagnostic_state"] = "wrong_object_control"
        elif decision.state == "object_lost":
            evidence["diagnostic_state"] = "object_loss_risk"
        elif decision.state == "no_progress" and metadata.get("phase_reliable") is True:
            evidence["diagnostic_state"] = {
                "approach": "approach_progress_stalled",
                "grasp": "grasp_progress_stalled",
                "place": "goal_relation_progress_stalled",
                "manipulate": "goal_relation_progress_stalled",
                "goal_relation": "goal_relation_progress_stalled",
            }.get(str(metadata.get("phase")), "task_progress_stall")
        if decision.state in {"goal_achieved", "progress"}:
            self.no_progress_run = 0
            return MonitorEvent(
                EventType.NORMAL, decision.score, decision.confidence, evidence,
                RecommendedAction.CONTINUE, self.name, action_index,
            )
        if decision.state in {"unknown", "ambiguous"}:
            self.no_progress_run = 0
            return MonitorEvent(
                EventType.OBJECT_AMBIGUOUS, decision.score, decision.confidence, evidence,
                RecommendedAction.REQUEST_MORE_EVIDENCE, self.name, action_index,
            )
        if decision.state == "no_progress":
            self.no_progress_run += 1
            evidence["no_progress_run"] = self.no_progress_run
            if self.no_progress_run < self.config.no_progress_confirmations:
                return MonitorEvent(
                    EventType.OBJECT_AMBIGUOUS, decision.score, decision.confidence, evidence,
                    RecommendedAction.REQUEST_MORE_EVIDENCE, self.name, action_index,
                )
        else:
            self.no_progress_run = 0
        if decision.state in self.config.immediate_states or decision.state == "no_progress":
            return MonitorEvent(
                EventType.OBJECT_FAILURE, decision.score, decision.confidence, evidence,
                RecommendedAction.STOP_AND_REPLAN, self.name, action_index,
            )
        return MonitorEvent(
            EventType.OBJECT_AMBIGUOUS, decision.score, decision.confidence, evidence,
            RecommendedAction.REQUEST_MORE_EVIDENCE, self.name, action_index,
        )


class CalibratedSemanticConsequenceAdapter:
    """Online adapter with an explicit information firewall and shadow mode.

    ``evidence_scorer`` receives paired images, the intended action and a
    history stripped of measured robot state/reward/success labels.  It must
    return :class:`SemanticConsequenceEvidence`; uncalibrated evidence is
    therefore forced to abstain by :class:`SemanticConsequenceGate`.
    """

    name = "calibrated_semantic_task_consequence"

    def __init__(self, evidence_scorer, *, gate: SemanticConsequenceGate | None = None,
                 temporal: SemanticTemporalConfig = SemanticTemporalConfig(),
                 control_enabled: bool = False):
        self.evidence_scorer = evidence_scorer
        self.gate = gate or SemanticConsequenceGate()
        self.monitor = SemanticConsequenceMonitor(temporal)
        self.control_enabled = bool(control_enabled)

    def reset(self):
        self.monitor.reset()
        reset = getattr(self.evidence_scorer, "reset", None)
        if reset is not None:
            reset()

    def observe_conditioned(self, *, upstream_events, images_before, images_after,
                            intended_action, history, action_index):
        safe_history = tuple({
            "action_index": item.get("action_index"),
            "intended_action": item.get("intended_action"),
        } for item in history)
        evidence = self.evidence_scorer(
            images_before, images_after, intended_action, safe_history)
        if not isinstance(evidence, SemanticConsequenceEvidence):
            raise TypeError("semantic evidence scorer must return SemanticConsequenceEvidence")
        decision = self.gate.decide(evidence)
        event = self.monitor.observe_decision(decision, action_index=action_index)
        enriched = {
            **dict(event.evidence),
            "semantic_control_enabled": self.control_enabled,
            "upstream_event_types": [item.event_type.value for item in upstream_events],
        }
        # Missing observability or calibration is a component-availability
        # fact, not evidence that robot motion should be slowed or truncated.
        # Keep it in the audit log without granting it any control authority.
        if decision.state == "unknown":
            return MonitorEvent(
                EventType.NORMAL, event.score, event.confidence, enriched,
                RecommendedAction.CONTINUE, self.name,
                event.action_index, event.timestamp)
        if self.control_enabled or event.event_type is not EventType.OBJECT_FAILURE:
            return MonitorEvent(
                event.event_type, event.score, event.confidence, enriched,
                event.recommended_action, self.name, event.action_index, event.timestamp)
        return MonitorEvent(
            EventType.OBJECT_AMBIGUOUS, event.score, event.confidence,
            {**enriched,
             "shadow_original_event_type": event.event_type.value,
             "shadow_original_recommended_action": event.recommended_action.value},
            RecommendedAction.REQUEST_MORE_EVIDENCE, self.name,
            event.action_index, event.timestamp)
