from __future__ import annotations

from typing import Any, Mapping, Protocol, Sequence

import numpy as np

from .events import MonitorEvent


State = Mapping[str, Any]
History = Sequence[Mapping[str, Any]]


class StepMonitor(Protocol):
    name: str

    def reset(self) -> None: ...

    def observe(self, *, intended_action: np.ndarray, state_before: State,
                state_after: State, history: History, action_index: int) -> MonitorEvent: ...


class ObjectResultMonitor(Protocol):
    name: str

    def reset(self) -> None: ...

    def observe(self, *, images_before: Mapping[str, np.ndarray] | None,
                images_after: Mapping[str, np.ndarray] | None,
                intended_action: np.ndarray, history: History,
                action_index: int) -> MonitorEvent: ...
