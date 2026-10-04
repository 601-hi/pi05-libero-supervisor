from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Sequence


class RecoveryHandoffState(str, Enum):
    IDLE = "idle"
    REQUESTED = "requested"
    ACTIVE = "active"
    REPLAN_PENDING = "replan_pending"
    SAFE_STOP = "safe_stop"


@dataclass(frozen=True)
class RecoveryHandoffRequest:
    reason: str
    confidence: float
    action_index: int
    history: tuple[Mapping, ...]


class RecoveryExecutorPort:
    """Environment-neutral ownership boundary for third-layer recovery.

    This class deliberately does not score collisions or generate robot motion.
    A simulator or real-robot adapter claims a request, obtains its own sensor
    evidence, and submits small action batches back through ``SupervisorRuntime``.
    Those actions therefore still pass the ordinary instruction guard.
    """

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.state = RecoveryHandoffState.IDLE
        self.request: RecoveryHandoffRequest | None = None

    def request_recovery(self, *, reason: str, confidence: float,
                         action_index: int, history: Sequence[Mapping]) -> bool:
        if self.state not in (RecoveryHandoffState.IDLE,
                              RecoveryHandoffState.REPLAN_PENDING):
            return False
        self.request = RecoveryHandoffRequest(
            reason=str(reason), confidence=float(confidence),
            action_index=int(action_index), history=tuple(history))
        self.state = RecoveryHandoffState.REQUESTED
        return True

    def claim(self) -> RecoveryHandoffRequest | None:
        if self.state is not RecoveryHandoffState.REQUESTED:
            return None
        self.state = RecoveryHandoffState.ACTIVE
        return self.request

    def recovery_completed(self, *, safe: bool, needs_replan: bool = True) -> None:
        if self.state is not RecoveryHandoffState.ACTIVE:
            raise RuntimeError("external recovery is not active")
        if not safe:
            self.state = RecoveryHandoffState.SAFE_STOP
        elif needs_replan:
            self.state = RecoveryHandoffState.REPLAN_PENDING
        else:
            self.state = RecoveryHandoffState.IDLE
            self.request = None

    def replan_completed(self) -> None:
        if self.state is RecoveryHandoffState.REPLAN_PENDING:
            self.state = RecoveryHandoffState.IDLE
            self.request = None

    @property
    def owns_control(self) -> bool:
        return self.state in (RecoveryHandoffState.REQUESTED,
                              RecoveryHandoffState.ACTIVE,
                              RecoveryHandoffState.REPLAN_PENDING,
                              RecoveryHandoffState.SAFE_STOP)


class HybridRecoveryExecutorAdapter(RecoveryExecutorPort):
    """Bind the generic handoff port to a ``HybridEscapeCoordinator``.

    Robot-specific code still owns sensing, candidate construction and action
    execution.  It calls the coordinator and reports each returned decision to
    :meth:`accept_coordinator_decision`; this adapter converts only terminal
    coordinator states into runtime ownership transitions.
    """

    def __init__(self, coordinator):
        self.coordinator = coordinator
        self.last_decision = None
        super().__init__()

    def reset(self) -> None:
        super().reset()
        self.last_decision = None

    def claim(self) -> RecoveryHandoffRequest | None:
        request = super().claim()
        if request is not None:
            self.accept_coordinator_decision(self.coordinator.trigger())
        return request

    def accept_coordinator_decision(self, decision):
        if self.state is not RecoveryHandoffState.ACTIVE:
            raise RuntimeError("coordinator decision outside active recovery")
        self.last_decision = decision
        if decision.state == "safe_stop":
            self.recovery_completed(safe=False)
        elif decision.state == "reobserve_and_replan":
            self.recovery_completed(safe=True, needs_replan=True)
        return decision

    def synchronize_terminal_state(self) -> RecoveryHandoffState:
        """Map the coordinator's current terminal state after a physical loop.

        Environment runners often receive many intermediate decisions.  This
        method avoids manufacturing a fake final decision merely to close the
        ownership handshake.
        """
        if self.state is not RecoveryHandoffState.ACTIVE:
            return self.state
        machine_state = self.coordinator.machine.state
        if machine_state == "safe_stop":
            self.recovery_completed(safe=False)
        elif machine_state == "reobserve_and_replan":
            self.recovery_completed(safe=True, needs_replan=True)
        return self.state
