"""Progress-only monitoring for articulated goal relations.

The monitor intentionally cannot declare a goal satisfied.  A perception
provider supplies a calibrated proxy measurement (for example a tracked open
region area); this module only detects causal stagnation or regression.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .events import EventType, MonitorEvent, RecommendedAction


@dataclass(frozen=True)
class ArticulationProgressConfig:
    calibration_id: str
    initial_grace_actions: int = 80
    active_gain: float = 0.15
    plateau_window_actions: int = 30
    refresh_gain: float = 0.03
    regression_drop: float = 0.25
    regression_confirmations: int = 5
    smoothing_samples: int = 5
    minimum_confidence: float = 0.70

    def __post_init__(self) -> None:
        if not self.calibration_id:
            raise ValueError("calibration_id is required")
        if self.initial_grace_actions < 1 or self.plateau_window_actions < 1:
            raise ValueError("action windows must be positive")
        if self.regression_confirmations < 1 or self.smoothing_samples < 1:
            raise ValueError("confirmation windows must be positive")
        for value in (self.active_gain, self.refresh_gain, self.regression_drop,
                      self.minimum_confidence):
            if not 0.0 <= value <= 1.0:
                raise ValueError("normalized thresholds must lie in [0, 1]")


@dataclass(frozen=True)
class ArticulationMeasurement:
    action_index: int
    area_fraction: float | None
    confidence: float
    observable: bool
    identity_reliable: bool
    calibrated: bool
    calibration_id: str
    relation_id: str
    predicate: str
    provenance: str = "unknown"


@dataclass(frozen=True)
class ArticulationProgressDecision:
    state: str
    action_index: int
    progress: float | None
    best_progress: float | None
    confidence: float
    reason: str


class ArticulationProgressWatchdog:
    def __init__(self, config: ArticulationProgressConfig):
        self.config = config
        self.reset()

    def reset(self) -> None:
        self._initial_area = None
        self._last_index = None
        self._raw = deque(maxlen=self.config.smoothing_samples)
        self._history: list[tuple[int, float, float]] = []
        self._best = 0.0
        self._regression_run = 0
        self._latched_state = None

    def update(self, measurement: ArticulationMeasurement) -> ArticulationProgressDecision:
        invalid = None
        if self._last_index is not None and measurement.action_index <= self._last_index:
            invalid = "measurement is stale or action index is not increasing"
        elif not measurement.observable:
            invalid = "articulation relation is not observable"
        elif not measurement.identity_reliable:
            invalid = "articulation identity is not reliable"
        elif not measurement.calibrated or measurement.calibration_id != self.config.calibration_id:
            invalid = "measurement is not covered by the frozen calibration"
        elif not 0.0 <= measurement.confidence <= 1.0 or measurement.confidence < self.config.minimum_confidence:
            invalid = "measurement confidence is below the frozen gate"
        elif (measurement.area_fraction is None
              or not np.isfinite(measurement.area_fraction)
              or measurement.area_fraction <= 0.0):
            invalid = "area proxy is invalid"
        if invalid:
            return ArticulationProgressDecision(
                "unknown", measurement.action_index, None, None, 0.0, invalid)

        self._last_index = measurement.action_index
        if self._initial_area is None:
            self._initial_area = float(measurement.area_fraction)
        raw_progress = 1.0 - float(measurement.area_fraction) / self._initial_area
        self._raw.append(raw_progress)
        progress = float(np.median(self._raw))
        previous_best = self._best
        self._best = max(self._best, progress)
        self._history.append((measurement.action_index, progress, self._best))

        if self._latched_state is not None:
            return ArticulationProgressDecision(
                self._latched_state, measurement.action_index, progress, self._best,
                measurement.confidence, "previous actionable relation event remains latched")

        if self._best - progress >= self.config.regression_drop:
            self._regression_run += 1
        else:
            self._regression_run = 0
        if self._regression_run >= self.config.regression_confirmations:
            self._latched_state = "goal_regression"
            return ArticulationProgressDecision(
                self._latched_state, measurement.action_index, progress, self._best,
                measurement.confidence, "best relation progress was lost persistently")

        elapsed = measurement.action_index - self._history[0][0]
        if elapsed >= self.config.initial_grace_actions and self._best < self.config.active_gain:
            self._latched_state = "goal_no_progress"
            return ArticulationProgressDecision(
                self._latched_state, measurement.action_index, progress, self._best,
                measurement.confidence, "no significant relation progress within grace budget")

        cutoff = measurement.action_index - self.config.plateau_window_actions
        old = [item for item in self._history if item[0] <= cutoff]
        if self._best >= self.config.active_gain and old:
            old_best = old[-1][2]
            if self._best - old_best < self.config.refresh_gain:
                self._latched_state = "goal_no_progress"
                return ArticulationProgressDecision(
                    self._latched_state, measurement.action_index, progress, self._best,
                    measurement.confidence, "relation progress plateaued after initial improvement")

        state = "progress" if self._best > previous_best else "observing"
        return ArticulationProgressDecision(
            state, measurement.action_index, progress, self._best,
            measurement.confidence, "relation proxy remains non-actionable")


class ArticulationProgressMonitor:
    """Conditioned runtime monitor backed by a pluggable perception provider."""

    name = "articulation_relation_progress"

    def __init__(self, provider, config: ArticulationProgressConfig, *, control_enabled=False):
        self.provider = provider
        self.watchdog = ArticulationProgressWatchdog(config)
        self.control_enabled = bool(control_enabled)

    def reset(self) -> None:
        self.watchdog.reset()
        reset = getattr(self.provider, "reset", None)
        if reset is not None:
            reset()

    def observe_conditioned(self, *, upstream_events, images_before, images_after,
                            intended_action, history, action_index) -> MonitorEvent:
        value = self.provider(images_before, images_after, intended_action, history, action_index)
        if not isinstance(value, ArticulationMeasurement):
            raise TypeError("articulation provider must return ArticulationMeasurement")
        decision = self.watchdog.update(value)
        diagnostic = {
            "goal_no_progress": "goal_relation_progress_stalled",
            "goal_regression": "goal_relation_not_satisfied",
        }.get(decision.state)
        evidence: dict[str, object] = {
            "task_relation_event": decision.state,
            "relation_id": value.relation_id,
            "predicate": value.predicate,
            "progress": decision.progress,
            "best_progress": decision.best_progress,
            "relation_reason": decision.reason,
            "calibration_id": value.calibration_id,
            "provenance": value.provenance,
            "phase": "goal_relation",
            "phase_reliable": value.identity_reliable,
            "semantic_control_enabled": self.control_enabled,
            "upstream_event_types": [event.event_type.value for event in upstream_events],
        }
        if diagnostic is not None:
            evidence["diagnostic_state"] = diagnostic
        if decision.state == "unknown":
            return MonitorEvent(EventType.NORMAL, 0.0, 0.0, evidence,
                                RecommendedAction.CONTINUE, self.name, action_index)
        if decision.state not in {"goal_no_progress", "goal_regression"}:
            return MonitorEvent(EventType.NORMAL, max(0.0, decision.progress or 0.0),
                                decision.confidence, evidence, RecommendedAction.CONTINUE,
                                self.name, action_index)
        actionable = MonitorEvent(
            EventType.OBJECT_FAILURE, 1.0, decision.confidence, evidence,
            RecommendedAction.STOP_AND_REPLAN, self.name, action_index)
        if self.control_enabled:
            return actionable
        return MonitorEvent(
            EventType.OBJECT_AMBIGUOUS, actionable.score, actionable.confidence,
            {**evidence, "shadow_original_event_type": EventType.OBJECT_FAILURE.value,
             "shadow_original_recommended_action": RecommendedAction.STOP_AND_REPLAN.value},
            RecommendedAction.REQUEST_MORE_EVIDENCE, self.name, action_index)
