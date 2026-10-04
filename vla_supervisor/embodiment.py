"""Small, explicit robot embodiment adapters for recovery primitives."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence
import numpy as np


class RecoveryEmbodiment(Protocol):
    """Translate generic recovery capabilities into robot actions."""
    adapter_id: str

    def hold_action(self, last_action: Sequence[float] | None) -> tuple[float, ...]: ...
    def retreat_action(self, last_action: Sequence[float] | None,
                       magnitude: float) -> tuple[float, ...] | None: ...
    def release_action(self, reference_action: Sequence[float] | None, *,
                       preserve_motion: bool = False) -> tuple[float, ...] | None: ...


@dataclass(frozen=True)
class CartesianDeltaEmbodiment:
    """Adapter for normalized Cartesian-delta actions such as LIBERO's.

    This is a carrier description, not a task policy. Other robots implement
    the same protocol using their controller semantics and safety limits.
    """
    action_dimension: int = 7
    translation_indices: tuple[int, int, int] = (0, 1, 2)
    gripper_index: int | None = 6
    adapter_id: str = 'cartesian-delta-v1'

    def __post_init__(self):
        if self.action_dimension < 1 or len(set(self.translation_indices)) != 3:
            raise ValueError('invalid action layout')
        indices = self.translation_indices + (() if self.gripper_index is None else (self.gripper_index,))
        if any(i < 0 or i >= self.action_dimension for i in indices):
            raise ValueError('action index outside declared dimension')

    def hold_action(self, last_action):
        size = max(self.action_dimension, len(last_action) if last_action is not None else 0)
        action = np.zeros(size, dtype=float)
        if last_action is not None and self.gripper_index is not None and len(last_action) > self.gripper_index:
            action[self.gripper_index] = float(last_action[self.gripper_index])
        return tuple(action.tolist())

    def retreat_action(self, last_action, magnitude):
        if last_action is None or len(last_action) < self.action_dimension:
            return None
        translation = np.asarray([last_action[i] for i in self.translation_indices], dtype=float)
        norm = float(np.linalg.norm(translation))
        if not np.isfinite(magnitude) or magnitude < 0 or not np.isfinite(norm) or norm < 1e-6:
            return None
        action = np.asarray(self.hold_action(last_action), dtype=float)
        reverse = -float(magnitude) * translation / norm
        for index, value in zip(self.translation_indices, reverse):
            action[index] = value
        return tuple(action.tolist())

    def release_action(self, reference_action, *, preserve_motion=False):
        if self.gripper_index is None:
            return None
        size = max(self.action_dimension, len(reference_action) if reference_action is not None else 0)
        if preserve_motion and reference_action is not None:
            action = np.zeros(size, dtype=float)
            action[:len(reference_action)] = np.asarray(reference_action, dtype=float)
        else:
            action = np.zeros(size, dtype=float)
        action[self.gripper_index] = -1.0
        return tuple(action.tolist())
