from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Mapping

import numpy as np

from .events import EventType, MonitorEvent, RecommendedAction


class NullExecutionMonitor:
    """Safe placeholder that declares execution scoring unavailable."""
    name = "execution_consistency_placeholder"
    def reset(self): pass
    def observe(self, *, intended_action, state_before, state_after, history, action_index):
        return MonitorEvent(EventType.NORMAL, 0.0, 0.0, {"status": "not_implemented"},
                            RecommendedAction.CONTINUE, self.name, action_index)


class DeterministicTestMonitor:
    """Explicit plumbing-only event injector; never enable for evaluation."""
    name = "test_injection"
    def __init__(self, event_type: EventType, start_action: int, num_steps: int,
                 confidence: float = 1.0):
        if event_type is EventType.NORMAL: raise ValueError("test event cannot be NORMAL")
        self.event_type=event_type;self.start_action=start_action;self.num_steps=num_steps;self.confidence=confidence
    def reset(self): pass
    def observe(self, *, intended_action, state_before, state_after, history, action_index):
        active=self.start_action <= action_index < self.start_action+self.num_steps
        return MonitorEvent(self.event_type if active else EventType.NORMAL,float(active),
                            self.confidence if active else 0.0,
                            {"test_only":True,"configured_start":self.start_action,"configured_steps":self.num_steps},
                            RecommendedAction.STOP_AND_REPLAN if active else RecommendedAction.CONTINUE,
                            self.name,action_index)


class CallbackExecutionMonitor:
    """Adapter for frozen normal/abnormal expert scorers."""
    name = "execution_consistency"

    def __init__(self, scorer: Callable[..., tuple[float, float, Mapping[str, Any]]], threshold: float,
                 minimum_confidence: float = .5):
        self.scorer, self.threshold = scorer, threshold
        self.minimum_confidence = float(minimum_confidence)

    def reset(self):
        reset = getattr(self.scorer, "reset", None)
        if reset is not None:
            reset()

    def observe(self, *, intended_action, state_before, state_after, history, action_index):
        score, confidence, evidence = self.scorer(intended_action, state_before, state_after, history)
        if score > self.threshold and confidence < self.minimum_confidence:
            return MonitorEvent(EventType.EXECUTION_AMBIGUOUS, float(score), float(confidence),
                                dict(evidence), RecommendedAction.REQUEST_MORE_EVIDENCE,
                                self.name, action_index)
        alarm = score > self.threshold and confidence >= self.minimum_confidence
        return MonitorEvent(EventType.EXECUTION_MISMATCH if alarm else EventType.NORMAL,
                            float(score), float(confidence), dict(evidence),
                            RecommendedAction.STOP_AND_REPLAN if alarm else RecommendedAction.CONTINUE,
                            self.name, action_index)


class FourStateExecutionMonitor:
    """Adapter from calibrated dual-expert output to supervisor events.

    The scorer returns a mapping with ``decision`` in
    {known_normal, known_abnormal, ambiguous_overlap, unknown}, ``lr`` and
    ``reliability``.  This preserves abstention instead of collapsing it into
    an arbitrary binary threshold.
    """

    name = "four_state_execution_consistency"
    valid_decisions = {"known_normal", "known_abnormal", "ambiguous_overlap", "unknown"}

    def __init__(self, scorer):
        self.scorer = scorer

    def reset(self):
        reset = getattr(self.scorer, "reset", None)
        if reset is not None:
            reset()

    def observe(self, *, intended_action, state_before, state_after, history, action_index):
        result = dict(self.scorer(intended_action, state_before, state_after, history))
        missing = {"decision", "lr", "reliability"}.difference(result)
        if missing:
            raise ValueError(f"four-state execution scorer missing fields: {sorted(missing)}")
        decision = str(result["decision"])
        if decision not in self.valid_decisions:
            raise ValueError(f"unsupported execution decision: {decision}")
        lr = float(result["lr"])
        reliability = float(np.clip(result["reliability"], 0.0, 1.0))
        if decision == "known_normal":
            kind, action = EventType.NORMAL, RecommendedAction.CONTINUE
        elif decision == "known_abnormal":
            kind, action = EventType.EXECUTION_MISMATCH, RecommendedAction.STOP_AND_REPLAN
        else:
            kind, action = EventType.EXECUTION_AMBIGUOUS, RecommendedAction.REQUEST_MORE_EVIDENCE
        return MonitorEvent(kind, lr, reliability, result, action, self.name, action_index)


class ContactAwareExecutionMonitor:
    """Route high execution scores using independent contact evidence.

    ``contact_scorer`` returns ``(probability, confidence, evidence)``. Contact
    is not inferred from the same command/EEF residual, which would provide no
    independent information and create circular gating.
    """
    name = "contact_aware_execution_consistency"

    def __init__(self, execution_scorer, execution_threshold: float, contact_scorer,
                 contact_threshold: float = .5, minimum_contact_confidence: float = .5):
        self.execution_scorer = execution_scorer
        self.execution_threshold = float(execution_threshold)
        self.contact_scorer = contact_scorer
        self.contact_threshold = float(contact_threshold)
        self.minimum_contact_confidence = float(minimum_contact_confidence)

    def reset(self):
        for scorer in (self.execution_scorer, self.contact_scorer):
            reset = getattr(scorer, "reset", None)
            if reset is not None: reset()

    def observe(self, *, intended_action, state_before, state_after, history, action_index):
        score, confidence, execution_evidence = self.execution_scorer(
            intended_action, state_before, state_after, history)
        if not (confidence >= .5 and score > self.execution_threshold):
            return MonitorEvent(EventType.NORMAL, float(score), float(confidence),
                                {"execution": dict(execution_evidence)},
                                RecommendedAction.CONTINUE, self.name, action_index)
        contact, contact_confidence, contact_evidence = self.contact_scorer(
            intended_action, state_before, state_after, history)
        evidence = {"execution": dict(execution_evidence),
                    "contact": dict(contact_evidence),
                    "contact_probability": float(contact),
                    "contact_confidence": float(contact_confidence)}
        independently_supported_contact = (
            contact_confidence >= self.minimum_contact_confidence and
            contact >= self.contact_threshold)
        if independently_supported_contact:
            return MonitorEvent(EventType.EXECUTION_AMBIGUOUS, float(score),
                                min(float(confidence), float(contact_confidence)), evidence,
                                RecommendedAction.REQUEST_MORE_EVIDENCE, self.name, action_index)
        return MonitorEvent(EventType.EXECUTION_MISMATCH, float(score), float(confidence), evidence,
                            RecommendedAction.STOP_AND_REPLAN, self.name, action_index)


@dataclass(frozen=True)
class StallConfig:
    window: int = 40
    gripper_reversal_threshold: int = 3
    low_progress_threshold: float = 0.05
    command_norm_threshold: float = 0.015
    low_progress_run_threshold: int = 12


class PolicyStallMonitor:
    name = "policy_stall"

    def __init__(self, config: StallConfig = StallConfig()):
        self.config = config; self.reset()

    def reset(self):
        self.previous_gripper = None; self.reversals = deque(); self.low_progress_run = 0

    def observe(self, *, intended_action, state_before, state_after, history, action_index):
        action = np.asarray(intended_action, float)
        grip = float(action[6]) if len(action) > 6 else 0.0
        reversal = (self.previous_gripper is not None and grip * self.previous_gripper < 0 and
                    abs(grip) > .5 and abs(self.previous_gripper) > .5)
        self.reversals.append(int(reversal))
        if len(self.reversals) > self.config.window: self.reversals.popleft()
        self.previous_gripper = grip
        command = .05 * np.clip(action[:3], -1, 1)
        p0, p1 = np.asarray(state_before["eef_pos"], float), np.asarray(state_after["eef_pos"], float)
        actual = p1 - p0; norm = float(np.linalg.norm(command))
        progress = float(command @ actual / (norm * norm + 1e-12))
        low = norm > self.config.command_norm_threshold and progress < self.config.low_progress_threshold
        self.low_progress_run = self.low_progress_run + 1 if low else 0
        reversal_count = sum(self.reversals)
        alarm = (reversal_count > self.config.gripper_reversal_threshold or
                 self.low_progress_run >= self.config.low_progress_run_threshold)
        evidence = {"gripper_reversals_window": reversal_count,
                    "low_progress_run": self.low_progress_run, "directed_progress": progress}
        score = max(reversal_count / max(self.config.gripper_reversal_threshold, 1),
                    self.low_progress_run / max(self.config.low_progress_run_threshold, 1))
        return MonitorEvent(EventType.POLICY_STALL if alarm else EventType.NORMAL, score,
                            min(1.0, score / 1.5), evidence,
                            RecommendedAction.STOP_AND_REPLAN if alarm else RecommendedAction.CONTINUE,
                            self.name, action_index)


class NullObjectResultMonitor:
    """Safe placeholder: never fabricates visual evidence."""
    name = "object_result_placeholder"
    def reset(self): pass
    def observe(self, *, images_before, images_after, intended_action, history, action_index):
        return MonitorEvent(EventType.NORMAL, 0.0, 0.0, {"status": "not_implemented"},
                            RecommendedAction.CONTINUE, self.name, action_index)


class CallbackObjectResultMonitor:
    name = "object_result"
    def __init__(self, scorer: Callable[..., tuple[float, float, Mapping[str, Any]]], threshold: float):
        self.scorer, self.threshold = scorer, threshold
    def reset(self): pass
    def observe(self, *, images_before, images_after, intended_action, history, action_index):
        # Information firewall: object vision cannot inspect measured robot
        # response, task identity, reward, success, or disturbance labels.
        safe_history = tuple({"action_index": item.get("action_index"),
                              "intended_action": item.get("intended_action")}
                             for item in history)
        score, confidence, evidence = self.scorer(images_before, images_after,
                                                  intended_action, safe_history)
        alarm = score > self.threshold
        return MonitorEvent(EventType.OBJECT_FAILURE if alarm else EventType.NORMAL, float(score),
                            float(confidence), dict(evidence),
                            RecommendedAction.STOP_AND_REPLAN if alarm else RecommendedAction.CONTINUE,
                            self.name, action_index)


class StructuredObjectResultMonitor:
    """Object-centric consequence state machine over independent vision output."""
    name = "object_consequence"
    def __init__(self, relation_scorer, visibility_threshold=.5, relation_threshold=.7):
        self.scorer=relation_scorer;self.visibility_threshold=float(visibility_threshold)
        self.relation_threshold=float(relation_threshold);self.reset()
    def reset(self):
        self.was_carrying=False
        reset=getattr(self.scorer,"reset",None)
        if reset is not None:reset()
    def observe(self, *, images_before, images_after, intended_action, history, action_index):
        safe_history=tuple({"action_index":x.get("action_index"),"intended_action":x.get("intended_action")} for x in history)
        relation=dict(self.scorer(images_before,images_after,np.asarray(intended_action,float),safe_history))
        required=("object_visible_prob","target_visible_prob","gripper_object_contact_prob",
                  "object_in_target_prob","release_prob","confidence")
        missing=[k for k in required if k not in relation]
        if missing:raise ValueError(f"object relation scorer missing fields: {missing}")
        confidence=float(relation["confidence"]);visible=min(float(relation["object_visible_prob"]),float(relation["target_visible_prob"]))
        if visible < self.visibility_threshold or confidence < .5:
            return MonitorEvent(EventType.OBJECT_AMBIGUOUS,1-visible,confidence,relation,
                                RecommendedAction.REQUEST_MORE_EVIDENCE,self.name,action_index)
        contact=float(relation["gripper_object_contact_prob"]);inside=float(relation["object_in_target_prob"]);release=float(relation["release_prob"])
        dropped=self.was_carrying and contact < 1-self.relation_threshold and inside < 1-self.relation_threshold and release < self.relation_threshold
        missed_target=release >= self.relation_threshold and inside < 1-self.relation_threshold
        self.was_carrying = contact >= self.relation_threshold
        failure=dropped or missed_target
        evidence={**relation,"failure_mechanism":"unexpected_drop" if dropped else "released_outside_target" if missed_target else "none"}
        return MonitorEvent(EventType.OBJECT_FAILURE if failure else EventType.NORMAL,float(max(1-inside,1-contact) if failure else 0),confidence,evidence,
                            RecommendedAction.STOP_AND_REPLAN if failure else RecommendedAction.CONTINUE,self.name,action_index)
