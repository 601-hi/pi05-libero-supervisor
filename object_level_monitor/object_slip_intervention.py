"""Controlled object-slip fault injection for LIBERO data collection.

This module is a fault generator, not part of the deployable monitor.  Object
identity and simulator arrays may be used here, but every such field must be
logged under an ``oracle_only_`` name and excluded from detector inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class SlipInterventionResult:
    active: bool
    body_id: int
    applied_force_world_n: tuple[float, float, float]
    friction_scale: float


class ObjectSlipIntervention:
    """Temporarily reduce target friction and optionally apply weight-scaled force."""

    def __init__(
        self,
        sim,
        *,
        root_body_name: str,
        contact_geom_names: Sequence[str],
        friction_scale: float = 0.0,
        downward_force_weight_ratio: float = 0.0,
        gravity_mps2: float = 9.81,
    ) -> None:
        if not 0.0 <= friction_scale <= 1.0:
            raise ValueError("friction_scale must be in [0, 1]")
        if downward_force_weight_ratio < 0.0:
            raise ValueError("downward_force_weight_ratio must be non-negative")
        self.sim = sim
        self.body_id = int(sim.model.body_name2id(root_body_name))
        self.geom_ids = np.asarray(
            [int(sim.model.geom_name2id(name)) for name in contact_geom_names], dtype=np.int64
        )
        if not len(self.geom_ids):
            raise ValueError("at least one contact geometry is required")
        self.friction_scale = float(friction_scale)
        self.force_ratio = float(downward_force_weight_ratio)
        self.gravity = float(gravity_mps2)
        self._original_friction = np.asarray(sim.model.geom_friction[self.geom_ids]).copy()
        self._active = False

    def apply(self, active: bool) -> SlipInterventionResult:
        force = np.zeros(3, dtype=np.float64)
        if active:
            self.sim.model.geom_friction[self.geom_ids] = (
                self._original_friction * self.friction_scale
            )
            mass = float(self.sim.model.body_mass[self.body_id])
            force[2] = -self.force_ratio * mass * self.gravity
            self.sim.data.xfrc_applied[self.body_id, :3] = force
            self._active = True
        else:
            if self._active:
                self.sim.model.geom_friction[self.geom_ids] = self._original_friction
            self.sim.data.xfrc_applied[self.body_id, :3] = 0.0
            self._active = False
        return SlipInterventionResult(
            active=bool(active),
            body_id=self.body_id,
            applied_force_world_n=tuple(float(value) for value in force),
            friction_scale=self.friction_scale if active else 1.0,
        )

    def restore(self) -> None:
        """Restore simulator parameters and clear external wrench."""
        self.apply(False)


class OracleLiftTrigger:
    """Fault-generator trigger based on privileged target height, never a monitor input."""

    def __init__(self, initial_height_m: float, rise_threshold_m: float = 0.03, consecutive: int = 2):
        if rise_threshold_m <= 0.0 or consecutive <= 0:
            raise ValueError("rise threshold and persistence must be positive")
        self.initial_height = float(initial_height_m)
        self.rise_threshold = float(rise_threshold_m)
        self.required = int(consecutive)
        self.run = 0
        self.triggered = False

    def update(self, target_height_m: float, gripper_close_intent: bool) -> bool:
        if self.triggered:
            return True
        lifted = float(target_height_m) - self.initial_height >= self.rise_threshold
        self.run = self.run + 1 if lifted and gripper_close_intent else 0
        if self.run >= self.required:
            self.triggered = True
        return self.triggered
