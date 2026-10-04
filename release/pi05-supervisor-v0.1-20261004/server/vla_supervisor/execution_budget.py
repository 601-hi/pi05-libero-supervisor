"""Recovery-aware policy execution budget.

Rollback actions have their own bounded budget.  This object controls only the
policy task window and deliberately refreshes it after a completed rollback,
when the policy is observing a materially new initial condition.
"""
from __future__ import annotations


class RecoveryAwarePolicyBudget:
    def __init__(self, maximum_policy_steps: int):
        if int(maximum_policy_steps) <= 0:
            raise ValueError("maximum_policy_steps must be positive")
        self.maximum_policy_steps = int(maximum_policy_steps)
        self.actions_since_reset = 0
        self.reset_count = 0

    @property
    def available(self) -> bool:
        return self.actions_since_reset < self.maximum_policy_steps

    def consume_policy_action(self) -> None:
        self.actions_since_reset += 1

    def reset_after_completed_recovery(self) -> None:
        self.actions_since_reset = 0
        self.reset_count += 1

    def should_continue(self, *, settling: bool,
                        external_recovery_active: bool) -> bool:
        return bool(settling or self.available or external_recovery_active)
