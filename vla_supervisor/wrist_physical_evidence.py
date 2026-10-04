"""Structured wrist-camera evidence without pretending scores are probabilities."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Iterable


class WristSelectionState(str, Enum):
    NO_CANDIDATE = "no_candidate"
    UNIQUE = "unique"
    AMBIGUOUS = "ambiguous"


@dataclass(frozen=True)
class WristCandidateScore:
    candidate_id: str
    physical_score: float
    postclose_visibility: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.postclose_visibility <= 1.0:
            raise ValueError("postclose_visibility must be in [0, 1]")


@dataclass(frozen=True)
class WristSelectionConfig:
    score_margin: float = 0.10
    minimum_visibility: float = 0.5

    def __post_init__(self) -> None:
        if self.score_margin < 0.0:
            raise ValueError("score_margin must be non-negative")
        if not 0.0 <= self.minimum_visibility <= 1.0:
            raise ValueError("minimum_visibility must be in [0, 1]")


@dataclass(frozen=True)
class WristPhysicalSelection:
    state: WristSelectionState
    candidate_ids: tuple[str, ...]
    top_score: float | None
    runner_up_gap: float | None
    # Populated only by a separately fitted and validated calibrator.
    calibrated_candidate_probability: float | None = None

    @property
    def calibrated_attachment_probability(self) -> float | None:
        """Compatibility alias; this value does not establish attachment."""
        return self.calibrated_candidate_probability


@dataclass(frozen=True)
class PlattCalibration:
    mean: float
    scale: float
    intercept: float
    coefficient: float

    def __post_init__(self) -> None:
        if self.scale <= 0.0:
            raise ValueError("calibration scale must be positive")

    def probability(self, score: float) -> float:
        logit = self.intercept + self.coefficient * ((score - self.mean) / self.scale)
        logit = max(-40.0, min(40.0, logit))
        return float(1.0 / (1.0 + math.exp(-logit)))


def select_wrist_candidates(
    candidates: Iterable[WristCandidateScore],
    config: WristSelectionConfig = WristSelectionConfig(),
    calibration: PlattCalibration | None = None,
) -> WristPhysicalSelection:
    eligible = sorted(
        (candidate for candidate in candidates if candidate.postclose_visibility >= config.minimum_visibility),
        key=lambda candidate: candidate.physical_score,
        reverse=True,
    )
    if not eligible:
        return WristPhysicalSelection(WristSelectionState.NO_CANDIDATE, (), None, None)
    top = eligible[0].physical_score
    selected = tuple(
        candidate.candidate_id for candidate in eligible
        if candidate.physical_score >= top - config.score_margin
    )
    gap = None if len(eligible) < 2 else float(top - eligible[1].physical_score)
    state = WristSelectionState.UNIQUE if len(selected) == 1 else WristSelectionState.AMBIGUOUS
    probability = calibration.probability(top) if calibration is not None and state is WristSelectionState.UNIQUE else None
    return WristPhysicalSelection(state, selected, float(top), gap, probability)
