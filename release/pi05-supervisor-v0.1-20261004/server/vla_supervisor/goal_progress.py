"""Causal, stage-conditioned goal progress measurements for shadow evaluation.

No outcome labels, simulator object coordinates, or model-generated action
commands are accepted. Thresholds must be supplied by a calibration artifact.
This module never issues a stop/replan: real visual localization is not yet
validated. A shadow candidate requests more evidence, not a failure label.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math

from .events import EventType, MonitorEvent, RecommendedAction


@dataclass(frozen=True)
class GoalProgressObservation:
    action_index: int
    phase: str
    target_id: str
    measurement_domain: str
    metric_kind: str
    goal_error: float | None
    error_bound: float | None
    eef_position: tuple[float, float, float]
    observable: bool
    identity_reliable: bool
    phase_reliable: bool
    # Domain binds camera/preprocessing and fixed normalization scale. Pixel
    # distances from a moving camera need compensation, not a renamed domain.
    domain_verified: bool
    geometry_compensated: bool = False


@dataclass(frozen=True)
class GoalProgressConfig:
    calibration_id: str
    measurement_domain: str
    metric_kind: str
    window_actions: int
    max_observation_gap: int
    minimum_samples: int
    minimum_path_m: float
    minimum_progress: float
    away_threshold: float
    goal_tolerance: float
    maximum_error_bound: float
    confirmations: int = 2

    def __post_init__(self):
        if not self.calibration_id or not self.measurement_domain or not self.metric_kind:
            raise ValueError('calibration and measurement provenance are required')
        if self.window_actions < 1 or self.max_observation_gap < 1 or self.minimum_samples < 2 or self.confirmations < 1:
            raise ValueError('invalid temporal configuration')
        for name in ('minimum_path_m', 'minimum_progress', 'away_threshold', 'goal_tolerance', 'maximum_error_bound'):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError('invalid threshold: ' + name)
        if self.minimum_progress == 0 or self.away_threshold == 0:
            raise ValueError('progress and away thresholds must be positive')


@dataclass(frozen=True)
class GoalProgressResult:
    state: str
    reason: str
    action_index: int
    confirmed: bool = False
    progress_lower: float | None = None
    progress_upper: float | None = None
    sampled_path_m: float | None = None
    action_span: int = 0
    confirmations: int = 0

    def shadow_event(self) -> MonitorEvent:
        # Even an apparent goal-band match is not task success. Goal relations
        # and device state must be checked by the semantic consequence branch.
        return MonitorEvent(
            EventType.OBJECT_AMBIGUOUS, 0.0, 0.0,
            {'goal_progress_state': self.state, 'reason': self.reason,
             'shadow_only': True, 'confirmed_candidate': self.confirmed,
             'progress_lower': self.progress_lower, 'progress_upper': self.progress_upper,
             'sampled_path_m': self.sampled_path_m, 'action_span': self.action_span},
            RecommendedAction.REQUEST_MORE_EVIDENCE, 'goal_progress_shadow', self.action_index,
        )


class GoalProgressMonitor:
    def __init__(self, config: GoalProgressConfig | None = None):
        self.config = config
        self.reset()

    def reset(self):
        self._last_index = None
        self._key = None
        self._history = deque()
        self._candidate = None
        self._confirmations = 0

    def _clear_evidence(self):
        self._history.clear()
        self._candidate = None
        self._confirmations = 0

    def observe(self, obs: GoalProgressObservation) -> GoalProgressResult:
        if self._last_index is not None and obs.action_index <= self._last_index:
            raise ValueError('action indices must increase; reset monitor between episodes')
        previous_index = self._last_index
        self._last_index = obs.action_index
        config = self.config
        invalid = None
        if config is None:
            invalid = 'no frozen goal-progress calibration'
        elif not (obs.observable and obs.identity_reliable and obs.phase_reliable and obs.domain_verified):
            invalid = 'visibility, identity, phase or camera domain is unverified'
        elif obs.measurement_domain != config.measurement_domain or obs.metric_kind != config.metric_kind:
            invalid = 'measurement domain or metric differs from calibration'
        elif not obs.target_id or obs.phase not in {'approach', 'transport', 'manipulate'}:
            invalid = 'target or operation phase unavailable'
        elif obs.metric_kind == 'image_distance' and not obs.geometry_compensated:
            invalid = 'raw moving-camera image distance is not goal progress'
        elif obs.phase == 'manipulate' and obs.metric_kind != 'task_state_error':
            invalid = 'contact operation needs state change, not proximity alone'
        elif (obs.goal_error is None or obs.error_bound is None
              or not math.isfinite(obs.goal_error) or not math.isfinite(obs.error_bound)
              or obs.goal_error < 0 or not 0 <= obs.error_bound <= config.maximum_error_bound
              or len(obs.eef_position) != 3 or not all(math.isfinite(x) for x in obs.eef_position)):
            invalid = 'invalid or too uncertain measurement'
        if invalid:
            self._clear_evidence()
            self._key = None
            return GoalProgressResult('unknown', invalid, obs.action_index)
        key = (obs.phase, obs.target_id, obs.measurement_domain, obs.metric_kind)
        if key != self._key or (previous_index is not None and obs.action_index-previous_index > config.max_observation_gap):
            self._clear_evidence()
        self._key = key
        self._history.append(obs)
        if obs.goal_error + obs.error_bound <= config.goal_tolerance:
            self._clear_evidence()
            return GoalProgressResult('within_goal_band', 'proximity/state band is not task completion', obs.action_index)
        first = self._history[0]
        span = obs.action_index-first.action_index
        if span < config.window_actions or len(self._history) < config.minimum_samples:
            return GoalProgressResult('warming', 'waiting for a same-target causal window', obs.action_index, action_span=span)
        path = sum(math.dist(a.eef_position, b.eef_position) for a, b in zip(self._history, list(self._history)[1:]))
        progress = first.goal_error-obs.goal_error
        bound = first.error_bound+obs.error_bound
        lower, upper = progress-bound, progress+bound
        if lower >= config.minimum_progress:
            state, reason = 'progress', 'error decreases beyond measurement uncertainty'
        elif path < config.minimum_path_m:
            state, reason = 'insufficient_motion', 'defer stationary/contact cases to execution and state evidence'
        elif upper <= -config.away_threshold:
            state, reason = 'away_candidate', 'goal error increases beyond uncertainty'
        elif upper < config.minimum_progress and lower > -config.away_threshold:
            state, reason = 'stall_candidate', 'motion with bounded but insufficient goal improvement'
        else:
            state, reason = 'unknown', 'uncertainty overlaps progress or away boundaries'
        if state in {'away_candidate', 'stall_candidate'}:
            self._confirmations = self._confirmations+1 if self._candidate == state else 1
            self._candidate = state
        else:
            self._candidate = None
            self._confirmations = 0
        result = GoalProgressResult(state, reason, obs.action_index,
            self._confirmations >= config.confirmations, lower, upper, path, span, self._confirmations)
        # Consecutive windows share one boundary observation, not all but one
        # samples. Repeated evaluation of an overlapping window cannot confirm.
        self._history.clear()
        self._history.append(obs)
        return result


class GoalProgressQuerySchedule:
    """Independent low-frequency evidence requests, even when execution is normal.

    One instance per episode. Sparse observations must be reflected in the
    monitor's max gap; sampled EEF path is only a lower bound on true path.
    """
    def __init__(self, period_actions: int = 10, minimum_gap: int = 3):
        if period_actions < 1 or not 1 <= minimum_gap <= period_actions:
            raise ValueError('invalid query schedule')
        self.period_actions = period_actions
        self.minimum_gap = minimum_gap
        self.reset()

    def reset(self):
        self.last_query = None
        self.last_action = None

    def due(self, action_index: int, *, execution_ambiguous=False, phase_changed=False) -> bool:
        if self.last_action is not None and action_index <= self.last_action:
            raise ValueError('action indices must increase')
        self.last_action = action_index
        gap = math.inf if self.last_query is None else action_index-self.last_query
        due = gap >= self.period_actions or (gap >= self.minimum_gap and (execution_ambiguous or phase_changed))
        if due:
            self.last_query = action_index
        return due
