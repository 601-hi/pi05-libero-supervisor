from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .events import EventType, RecommendedAction, SupervisorDecision


class RecoveryState(str, Enum):
    RUNNING = "running"
    REPLAN_PENDING = "replan_pending"
    COOLDOWN = "cooldown"
    SAFE_STOPPED = "safe_stopped"


@dataclass(frozen=True)
class RecoveryConfig:
    max_replans: int = 3
    cooldown_steps: int = 5


class RecoveryController:
    def __init__(self, config: RecoveryConfig = RecoveryConfig()): self.config=config; self.reset()
    def reset(self): self.state=RecoveryState.RUNNING; self.replans=0; self.cooldown=0

    def apply(self, decision: SupervisorDecision) -> SupervisorDecision:
        if self.state is RecoveryState.SAFE_STOPPED:
            return SupervisorDecision(RecommendedAction.SAFE_STOP, decision.reason, decision.confidence)
        if self.state is RecoveryState.REPLAN_PENDING and decision.request_replan:
            # The bridge belongs to the recovery already counted. Its small or
            # reversed motion can keep the original detector active, but that
            # is not a new failed replan until a fresh policy chunk is installed.
            return SupervisorDecision(RecommendedAction.CONTINUE, decision.reason,
                                      decision.confidence, decision.triggering_events)
        if (self.state is RecoveryState.COOLDOWN and decision.request_replan and
                decision.reason not in {EventType.INSTRUCTION_UNSAFE, EventType.EXECUTION_MISMATCH}):
            # Suppress repeated semantic/stall replans while the new plan has
            # not had time to act.  Safety-relevant execution faults still pass.
            return SupervisorDecision(RecommendedAction.CONTINUE, decision.reason, decision.confidence,
                                      decision.triggering_events, cooldown_remaining=self.cooldown)
        if decision.request_replan:
            if self.replans >= self.config.max_replans:
                self.state=RecoveryState.SAFE_STOPPED
                return SupervisorDecision(RecommendedAction.SAFE_STOP, decision.reason, decision.confidence,
                                          decision.triggering_events, True, False)
            self.replans += 1; self.state=RecoveryState.REPLAN_PENDING
            return decision
        return decision

    def replan_completed(self):
        if self.state is RecoveryState.REPLAN_PENDING:
            self.cooldown=self.config.cooldown_steps
            self.state=RecoveryState.COOLDOWN if self.cooldown > 0 else RecoveryState.RUNNING

    def tick(self):
        if self.state is RecoveryState.COOLDOWN:
            self.cooldown=max(0,self.cooldown-1)
            if self.cooldown==0:self.state=RecoveryState.RUNNING
