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


class SupervisorRuntime:
    def __init__(self, instruction_guard: InstructionGuard, step_monitors, object_monitor,
                 fusion: TemporalFusion, recovery: RecoveryController,
                 logger: Utf8JsonlLogger | None = None, history_limit: int = 200,
                 intervention: InterventionPlanner | None = None,
                 control_enabled: bool = True):
        self.guard=instruction_guard;self.step_monitors=list(step_monitors);self.object_monitor=object_monitor
        self.fusion=fusion;self.recovery=recovery;self.logger=logger;self.history=deque(maxlen=history_limit)
        self.intervention=intervention or InterventionPlanner()
        # False makes the runtime a strictly observational shadow monitor.  It
        # may score and log hypothetical interventions, but must not mutate the
        # policy queue, recovery state, or replanning cadence.
        self.control_enabled=bool(control_enabled)
        self.chunk=deque();self.previous_action=None;self.last_intervention=None;self.recovery_bridge_active=False

    def reset_episode(self):
        self.chunk.clear();self.previous_action=None;self.last_intervention=None;self.recovery_bridge_active=False;self.history.clear();self.fusion.reset();self.recovery.reset()
        for monitor in [*self.step_monitors,self.object_monitor]: monitor.reset()

    def install_chunk(self, actions: Sequence[Sequence[float]]):
        self.chunk=deque(np.asarray(x,float) for x in actions)
        if self.logger:self.logger.write("chunk_installed",{"actions":len(self.chunk)})

    def stage_intervention_bridge(self) -> bool:
        """Install the latest bounded bridge actions into the guarded queue."""
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
        if self.recovery.state is not RecoveryState.REPLAN_PENDING or self.chunk:
            return False
        if self.last_intervention is None:
            return True
        return self.last_intervention.replan_after_bridge

    @property
    def recommended_chunk_horizon(self) -> int:
        return (self.last_intervention.next_chunk_horizon if self.last_intervention is not None
                else self.intervention.config.nominal_horizon)

    def next_action(self, state: Mapping, action_index: int):
        if not self.chunk:return None,SupervisorDecision(RecommendedAction.STOP_CHUNK,EventType.NORMAL,1.0)
        candidate=self.chunk[0];event=self.guard.check(candidate,state,self.previous_action,action_index)
        # The guard runs before execution and is deterministic/immediate.  A
        # normal guard pass must not advance post-step temporal monitor windows.
        if event.event_type is EventType.INSTRUCTION_UNSAFE:
            decision=SupervisorDecision(RecommendedAction.STOP_AND_REPLAN,event.event_type,event.confidence,
                                        (event,),clear_remaining_chunk=True,request_replan=True)
            decision=self.recovery.apply(decision)
        else:
            decision=SupervisorDecision(RecommendedAction.CONTINUE,EventType.NORMAL,1.0)
        if self.control_enabled and decision.clear_remaining_chunk:self.chunk.clear()
        if decision.action is not RecommendedAction.CONTINUE:
            self.last_intervention=self.intervention.plan(decision,[event],self.previous_action)
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
        decision=self.fusion.update(events)
        if self.control_enabled:
            decision=self.recovery.apply(decision)
        if self.control_enabled and decision.clear_remaining_chunk:self.chunk.clear()
        self.last_intervention=self.intervention.plan(decision,events,intended_action)
        if self.control_enabled and self.last_intervention.truncate_remaining_to is not None:
            keep=max(0,int(self.last_intervention.truncate_remaining_to))
            self.chunk=deque(islice(self.chunk,keep))
        self.history.append({"action_index":action_index,"intended_action":np.asarray(intended_action,float).tolist(),
                             "state_before":dict(state_before),"state_after":dict(state_after),
                             "events":[x.to_dict() for x in events]})
        self._log_decision(action_index,events,decision)
        if self.control_enabled:self.recovery.tick()
        return decision

    def mark_replan_completed(self, new_chunk):
        self.install_chunk(new_chunk);self.recovery_bridge_active=False;self.fusion.reset();self.recovery.replan_completed()
        for monitor in [*self.step_monitors,self.object_monitor]:monitor.reset()

    def _log_decision(self, action_index, events, decision):
        if self.logger:self.logger.write("supervisor_decision",{"action_index":action_index,
                            "events":[x.to_dict() for x in events],"decision":decision.to_dict(),
                            "remaining_chunk_actions":len(self.chunk),"recovery_state":self.recovery.state.value,
                            "replan_count":self.recovery.replans,
                            "intervention":None if self.last_intervention is None else self.last_intervention.to_dict()})
