"""Embodiment- and task-name-agnostic progress/budget assessment.

The assessor consumes an already validated scalar relation error (lower is
better). It does not localize objects, infer task success, or command recovery.
"""
from dataclasses import dataclass
import math

from .events import EventType, MonitorEvent, RecommendedAction


@dataclass(frozen=True)
class ProgressEnvelope:
    calibration_id: str
    relation_type: str
    minimum_progress_per_action: float
    minimum_path_efficiency: float
    goal_band: float
    maximum_stage_actions: int
    minimum_path: float

    def __post_init__(self):
        if not self.calibration_id or not self.relation_type:
            raise ValueError('frozen calibration provenance is required')
        values=(self.minimum_progress_per_action,self.minimum_path_efficiency,
                self.goal_band,self.minimum_path)
        if any(not math.isfinite(v) or v < 0 for v in values) or self.maximum_stage_actions < 1:
            raise ValueError('invalid progress envelope')


@dataclass(frozen=True)
class ProgressWindow:
    relation_type: str
    start_error: float
    end_error: float
    start_error_bound: float
    end_error_bound: float
    path_length: float
    action_span: int
    stage_elapsed_actions: int
    remaining_action_budget: int
    measurement_trusted: bool


@dataclass(frozen=True)
class ProgressBudgetAssessment:
    state: str
    reason: str
    progress_lower: float | None = None
    progress_upper: float | None = None
    efficiency_lower: float | None = None
    optimistic_actions_to_goal: float | None = None
    shadow_only: bool = True

    def to_shadow_event(self, *, action_index: int, source: str = 'progress_budget_shadow'):
        evidence = {
            'progress_budget_state': self.state, 'reason': self.reason,
            'progress_lower': self.progress_lower, 'progress_upper': self.progress_upper,
            'efficiency_lower': self.efficiency_lower,
            'optimistic_actions_to_goal': self.optimistic_actions_to_goal,
            'shadow_only': True,
        }
        if self.state in {'on_track', 'within_relation_band'}:
            return MonitorEvent(EventType.NORMAL, 0.0, 0.0, evidence,
                RecommendedAction.CONTINUE, source, action_index)
        kind = (EventType.RELATION_AMBIGUOUS if self.state in {'unknown','insufficient_motion'}
                else EventType.RELATION_PROGRESS_INADEQUATE)
        # No automatic recovery until end-to-end calibration is frozen.
        return MonitorEvent(kind, 0.0, 0.0, evidence,
            RecommendedAction.REQUEST_MORE_EVIDENCE, source, action_index)


def assess_progress_budget(window: ProgressWindow, envelope: ProgressEnvelope | None):
    unknown=lambda reason:ProgressBudgetAssessment('unknown',reason)
    if envelope is None:
        return unknown('no frozen success-trajectory envelope')
    if not window.measurement_trusted or window.relation_type != envelope.relation_type:
        return unknown('untrusted measurement or relation-envelope mismatch')
    values=(window.start_error,window.end_error,window.start_error_bound,
            window.end_error_bound,window.path_length)
    if (any(not math.isfinite(v) or v < 0 for v in values) or window.action_span < 1
            or window.stage_elapsed_actions < 0 or window.remaining_action_budget < 0):
        return unknown('invalid causal window')
    bound=window.start_error_bound+window.end_error_bound
    center=window.start_error-window.end_error
    lower,upper=center-bound,center+bound
    if window.end_error+window.end_error_bound <= envelope.goal_band:
        return ProgressBudgetAssessment('within_relation_band',
            'relation error is within its calibrated band, not necessarily task success',lower,upper)
    optimistic_rate=max(0.0,upper/window.action_span)
    optimistic_eta=(max(0.0,window.end_error-envelope.goal_band)/optimistic_rate
                    if optimistic_rate>0 else math.inf)
    efficiency=lower/max(window.path_length,1e-12)
    base=dict(progress_lower=lower,progress_upper=upper,
              efficiency_lower=efficiency,optimistic_actions_to_goal=optimistic_eta)
    if window.stage_elapsed_actions > envelope.maximum_stage_actions:
        return ProgressBudgetAssessment('stage_stalled','successful calibration envelope duration exceeded',**base)
    if optimistic_eta > window.remaining_action_budget:
        return ProgressBudgetAssessment('budget_infeasible','even optimistic recent progress cannot reach the relation band in budget',**base)
    if window.path_length < envelope.minimum_path:
        return ProgressBudgetAssessment('insufficient_motion','motion evidence is too small for path-efficiency diagnosis',**base)
    required=envelope.minimum_progress_per_action*window.action_span
    if lower >= required and efficiency >= envelope.minimum_path_efficiency:
        return ProgressBudgetAssessment('on_track','progress rate and path efficiency clear the frozen envelope',**base)
    if upper <= 0:
        return ProgressBudgetAssessment('wandering_or_away','motion does not reduce relation error beyond uncertainty',**base)
    return ProgressBudgetAssessment('slow_or_inefficient','some improvement is possible but it does not clear the frozen success envelope',**base)
