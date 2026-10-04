from __future__ import annotations

from collections import deque
from itertools import islice
from typing import Mapping, Sequence

import numpy as np

from .events import EventType, RecommendedAction, SupervisorDecision
from .fusion import TemporalFusion
from .instruction_guard import InstructionGuard
from .jsonl_log import Utf8JsonlLogger
from .recovery import RecoveryController
from .recovery import RecoveryState
from .intervention import InterventionPlanner
from .recovery_handoff import RecoveryExecutorPort, RecoveryHandoffState


class SupervisorRuntime:
    def __init__(self, instruction_guard: InstructionGuard, step_monitors, object_monitor,
                 fusion: TemporalFusion, recovery: RecoveryController,
                 logger: Utf8JsonlLogger | None = None, history_limit: int = 200,
                 intervention: InterventionPlanner | None = None,
                 control_enabled: bool = True, conditioned_monitors=(),
                 recovery_executor: RecoveryExecutorPort | None = None,
                 fusion_event_sources: set[str] | None = None):
        self.guard=instruction_guard;self.step_monitors=list(step_monitors);self.object_monitor=object_monitor
        self.fusion=fusion;self.recovery=recovery;self.logger=logger;self.history=deque(maxlen=history_limit)
        self.intervention=intervention or InterventionPlanner()
        # Optional post-step monitors consume images plus the already emitted
        # events.  This keeps task-consequence diagnosis independent from the
        # legacy single object-monitor slot while preserving compatibility.
        self.conditioned_monitors=list(conditioned_monitors)
        # False makes the runtime a strictly observational shadow monitor.  It
        # may score and log hypothetical interventions, but must not mutate the
        # policy queue, recovery state, or replanning cadence.
        self.control_enabled=bool(control_enabled)
        # Optional causal-isolation gate for component-level control pilots.
        # All monitor events are still logged, but only the named sources may
        # affect fusion, queue mutation, or recovery planning.
        self.fusion_event_sources=(None if fusion_event_sources is None else
                                   frozenset(str(x) for x in fusion_event_sources))
        self.recovery_executor=recovery_executor
        self.chunk=deque();self.previous_action=None;self.last_intervention=None;self.recovery_bridge_active=False
        self.novelty_only_steps_remaining=0;self.last_observation_novelty_only=False

    def reset_episode(self):
        self.chunk.clear();self.previous_action=None;self.last_intervention=None;self.recovery_bridge_active=False;self.history.clear();self.fusion.reset();self.recovery.reset()
        self.novelty_only_steps_remaining=0;self.last_observation_novelty_only=False
        for monitor in [*self.step_monitors,self.object_monitor,*self.conditioned_monitors]: monitor.reset()
        if self.recovery_executor is not None:self.recovery_executor.reset()

    def install_chunk(self, actions: Sequence[Sequence[float]]):
        self.chunk=deque(np.asarray(x,float) for x in actions)
        if self.logger:self.logger.write("chunk_installed",{"actions":len(self.chunk)})

    def begin_novelty_only_window(self, steps: int) -> None:
        """Keep post-replan monitors observable without letting them retake control."""
        self.novelty_only_steps_remaining=max(0,int(steps))

    def stage_intervention_bridge(self) -> bool:
        """Install the latest bounded bridge actions into the guarded queue."""
        if self.recovery_executor is not None and self.recovery_executor.owns_control:
            return False
        if (self.recovery_bridge_active or self.last_intervention is None or
                not self.last_intervention.bridge_actions):
            return False
        self.install_chunk(self.last_intervention.bridge_actions)
        self.recovery_bridge_active=True
        if self.logger:self.logger.write("intervention_bridge_staged",{
            "mode":self.last_intervention.mode.value,"actions":len(self.chunk),
            "replan_after_bridge":self.last_intervention.replan_after_bridge})
        return True

    @property
    def needs_policy_replan(self) -> bool:
        if self.recovery_executor is not None and self.recovery_executor.owns_control:
            return (self.recovery_executor.state is RecoveryHandoffState.REPLAN_PENDING
                    and not self.chunk)
        if self.recovery.state is not RecoveryState.REPLAN_PENDING or self.chunk:
            return False
        if self.last_intervention is None:
            return True
        return self.last_intervention.replan_after_bridge

    @property
    def recommended_chunk_horizon(self) -> int:
        return (self.last_intervention.next_chunk_horizon if self.last_intervention is not None
                else self.intervention.config.nominal_horizon)

    def build_policy_prompt(self, base_prompt: str, *, enabled: bool,
                            fallback_suffix: str = "") -> tuple[str, bool, str]:
        """Append recovery context only to the policy query that completes a replan.

        The online loop calls this immediately before inference and then calls
        :meth:`mark_replan_completed`.  Initial/nominal queries and disabled
        recovery prompting remain byte-for-byte unchanged.
        """
        directive = self.last_intervention
        pending = (
            enabled
            and self.recovery.state is RecoveryState.REPLAN_PENDING
            and directive is not None
            and directive.replan_after_bridge
        )
        if not pending:
            return str(base_prompt), False, "none"
        suffix = directive.recovery_prompt_suffix.strip() or str(fallback_suffix).strip()
        if not suffix:
            return str(base_prompt), False, directive.recovery_mechanism
        return f"{str(base_prompt).rstrip()}\n\n{suffix}", True, directive.recovery_mechanism

    def next_action(self, state: Mapping, action_index: int):
        if (self.control_enabled and self.recovery_executor is not None and
                self.recovery_executor.state is RecoveryHandoffState.SAFE_STOP):
            return None,SupervisorDecision(RecommendedAction.SAFE_STOP,EventType.EXECUTION_MISMATCH,1.0)
        if not self.chunk:return None,SupervisorDecision(RecommendedAction.STOP_CHUNK,EventType.NORMAL,1.0)
        candidate=self.chunk[0];event=self.guard.check(candidate,state,self.previous_action,action_index)
        # The guard runs before execution and is deterministic/immediate.  A
        # normal guard pass must not advance post-step temporal monitor windows.
        if event.event_type is EventType.INSTRUCTION_UNSAFE:
            decision=SupervisorDecision(RecommendedAction.STOP_AND_REPLAN,event.event_type,event.confidence,
                                        (event,),clear_remaining_chunk=True,request_replan=True)
            external_active = (
                self.control_enabled and self.recovery_executor is not None and
                self.recovery_executor.state is RecoveryHandoffState.ACTIVE)
            if external_active:
                # Recovery motion is never allowed to inherit the legacy
                # "duplicate alarm while replan pending" suppression.  An
                # unsafe recovery proposal is a hard boundary violation.
                self.recovery_executor.recovery_completed(safe=False)
                decision=SupervisorDecision(
                    RecommendedAction.SAFE_STOP,event.event_type,event.confidence,
                    (event,),clear_remaining_chunk=True,request_replan=False)
            else:
                decision=self.recovery.apply(decision)
        else:
            decision=SupervisorDecision(RecommendedAction.CONTINUE,EventType.NORMAL,1.0)
        if self.control_enabled and decision.clear_remaining_chunk:self.chunk.clear()
        if decision.action is not RecommendedAction.CONTINUE:
            self.last_intervention=self.intervention.plan(decision,[event],self.previous_action)
            self._request_external_recovery(decision,action_index)
            self._log_decision(action_index,[event],decision)
            if self.control_enabled:
                return None,decision
        action=self.chunk.popleft();self.previous_action=action.copy();return action,decision

    def observe_step(self, *, intended_action, state_before, state_after, action_index,
                     images_before=None, images_after=None):
        events=[m.observe(intended_action=np.asarray(intended_action,float),state_before=state_before,
                          state_after=state_after,history=tuple(self.history),action_index=action_index)
                for m in self.step_monitors]
        object_kwargs = dict(images_before=images_before, images_after=images_after,
                             intended_action=np.asarray(intended_action,float),
                             history=tuple(self.history), action_index=action_index)
        conditioned = getattr(self.object_monitor, "observe_conditioned", None)
        if conditioned is not None:
            events.append(conditioned(upstream_events=tuple(events), **object_kwargs))
        else:
            events.append(self.object_monitor.observe(**object_kwargs))
        for monitor in self.conditioned_monitors:
            observe = getattr(monitor, "observe_conditioned", None)
            if observe is None:
                raise TypeError("conditioned monitor must define observe_conditioned")
            events.append(observe(upstream_events=tuple(events), **object_kwargs))
        control_events = (
            events if self.fusion_event_sources is None else
            [event for event in events if event.source in self.fusion_event_sources]
        )
        decision=self.fusion.update(control_events)
        novelty_only=self.novelty_only_steps_remaining > 0
        self.last_observation_novelty_only=novelty_only
        if self.control_enabled and not novelty_only:
            decision=self.recovery.apply(decision)
        if self.control_enabled and not novelty_only and decision.clear_remaining_chunk:self.chunk.clear()
        self.last_intervention=self.intervention.plan(
            decision, control_events, intended_action)
        if not novelty_only and decision.action is not RecommendedAction.CONTINUE:
            self._request_external_recovery(decision,action_index)
        if (self.control_enabled and not novelty_only
                and self.last_intervention.truncate_remaining_to is not None):
            keep=max(0,int(self.last_intervention.truncate_remaining_to))
            self.chunk=deque(islice(self.chunk,keep))
        self.history.append({"action_index":action_index,"intended_action":np.asarray(intended_action,float).tolist(),
                             "state_before":dict(state_before),"state_after":dict(state_after),
                             "events":[x.to_dict() for x in events]})
        self._log_decision(action_index,events,decision)
        if self.control_enabled and not novelty_only:self.recovery.tick()
        if novelty_only:self.novelty_only_steps_remaining-=1
        return decision

    def mark_replan_completed(self, new_chunk):
        self.install_chunk(new_chunk);self.recovery_bridge_active=False;self.fusion.reset();self.recovery.replan_completed()
        if self.recovery_executor is not None:self.recovery_executor.replan_completed()
        for monitor in [*self.step_monitors,self.object_monitor,*self.conditioned_monitors]:monitor.reset()

    def claim_external_recovery(self):
        """Transfer control to the configured recovery executor, if requested."""
        if not self.control_enabled or self.recovery_executor is None:
            return None
        return self.recovery_executor.claim()

    def install_external_recovery_chunk(self, actions) -> bool:
        """Stage recovery actions; ``next_action`` still applies the guard."""
        if (not self.control_enabled or self.recovery_executor is None or
                self.recovery_executor.state is not RecoveryHandoffState.ACTIVE):
            return False
        self.install_chunk(actions)
        return True

    def finish_external_recovery(self, *, safe: bool, needs_replan: bool = True) -> None:
        if not self.control_enabled or self.recovery_executor is None:
            raise RuntimeError("external recovery control is not enabled")
        self.chunk.clear()
        self.recovery_executor.recovery_completed(safe=safe, needs_replan=needs_replan)

    def _request_external_recovery(self, decision, action_index: int) -> bool:
        if (not self.control_enabled or self.recovery_executor is None or
                not decision.clear_remaining_chunk):
            return False
        return self.recovery_executor.request_recovery(
            reason=decision.reason.value, confidence=decision.confidence,
            action_index=action_index, history=tuple(self.history))

    def _log_decision(self, action_index, events, decision):
        if self.logger:self.logger.write("supervisor_decision",{"action_index":action_index,
                            "events":[x.to_dict() for x in events],"decision":decision.to_dict(),
                            "remaining_chunk_actions":len(self.chunk),"recovery_state":self.recovery.state.value,
                            "replan_count":self.recovery.replans,
                            "novelty_only_control_suppressed":self.last_observation_novelty_only,
                            "recovery_handoff_state":None if self.recovery_executor is None else self.recovery_executor.state.value,
                            "intervention":None if self.last_intervention is None else self.last_intervention.to_dict()})
