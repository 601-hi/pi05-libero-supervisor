#!/usr/bin/env python3
"""Calibrated reliability, abstention, and causal sequential risk accumulation.

The module is model-agnostic. Calibration arrays must come from source-domain
held-out data and must be frozen before a target holdout is opened.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Literal

import numpy as np

Decision = Literal["known_normal", "known_abnormal", "ambiguous_overlap", "unknown"]


def _as_sorted_finite(values: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(values, dtype=float).reshape(-1)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError(f"{name} must contain finite calibration values")
    return np.sort(values)


def empirical_cdf(sorted_reference: np.ndarray, value: float) -> float:
    """Smoothed empirical CDF in (0, 1), robust at reference extremes."""
    n = len(sorted_reference)
    return float((np.searchsorted(sorted_reference, value, side="right") + 0.5) / (n + 1.0))


@dataclass(frozen=True)
class ReliabilityCalibration:
    # best_logp=max(log p_N, log p_A); low values indicate neither expert supports the sample.
    best_logp_reference: np.ndarray
    # Across-bootstrap standard deviation; high values indicate epistemic instability.
    ensemble_std_reference: np.ndarray
    normal_lr_threshold: float
    abnormal_lr_threshold: float
    minimum_reliability: float = 0.20
    minimum_sensor_quality: float = 0.50

    def __post_init__(self):
        object.__setattr__(self, "best_logp_reference", _as_sorted_finite(self.best_logp_reference, "best_logp_reference"))
        object.__setattr__(self, "ensemble_std_reference", _as_sorted_finite(self.ensemble_std_reference, "ensemble_std_reference"))
        if self.normal_lr_threshold >= self.abnormal_lr_threshold:
            raise ValueError("normal_lr_threshold must be below abnormal_lr_threshold")
        if not 0 <= self.minimum_reliability <= 1 or not 0 <= self.minimum_sensor_quality <= 1:
            raise ValueError("quality thresholds must be in [0, 1]")

    def score(self, normal_logp: float, abnormal_logp: float, ensemble_std: float, sensor_quality: float):
        if not np.isfinite([normal_logp, abnormal_logp, ensemble_std, sensor_quality]).all():
            return {"reliability": 0.0, "support": 0.0, "stability": 0.0, "sensor": 0.0, "decision": "unknown", "lr": float("nan")}
        sensor = float(np.clip(sensor_quality, 0.0, 1.0))
        best_logp = max(normal_logp, abnormal_logp)
        support = empirical_cdf(self.best_logp_reference, best_logp)
        # One minus percentile: a disagreement no larger than most calibration
        # disagreements remains reliable; extreme disagreement approaches zero.
        stability = 1.0 - empirical_cdf(self.ensemble_std_reference, ensemble_std)
        reliability = float(min(support, stability, sensor))
        lr = float(abnormal_logp - normal_logp)
        if sensor < self.minimum_sensor_quality or reliability < self.minimum_reliability:
            decision: Decision = "unknown"
        elif lr <= self.normal_lr_threshold:
            decision = "known_normal"
        elif lr >= self.abnormal_lr_threshold:
            decision = "known_abnormal"
        else:
            decision = "ambiguous_overlap"
        return {"reliability": reliability, "support": support, "stability": stability, "sensor": sensor, "decision": decision, "lr": lr}

    @classmethod
    def from_json(cls, path: str | Path, minimum_reliability: float = 0.20, minimum_sensor_quality: float = 0.50):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not payload.get("source_only"):
            raise ValueError("reliability calibration is not marked source_only")
        if payload.get("overlap_warning"):
            raise ValueError("normal and abnormal likelihood-ratio gates overlap; calibration is not deployable")
        return cls(
            best_logp_reference=np.asarray(payload["best_logp_reference"]),
            ensemble_std_reference=np.asarray(payload["ensemble_std_reference"]),
            normal_lr_threshold=float(payload["normal_lr_threshold"]),
            abnormal_lr_threshold=float(payload["abnormal_lr_threshold"]),
            minimum_reliability=minimum_reliability,
            minimum_sensor_quality=minimum_sensor_quality,
        )


@dataclass
class SensorQualityMonitor:
    expected_dt: float
    dt_tolerance_fraction: float = 0.25
    max_position_jump: float = 0.10
    max_joint_speed: float = 5.0

    def score(self, dt: float, eef_delta_norm: float, joint_speed_norm: float, finite: bool = True) -> float:
        if not finite or not np.isfinite([dt, eef_delta_norm, joint_speed_norm]).all() or dt <= 0:
            return 0.0
        timing_error = abs(dt - self.expected_dt) / max(self.expected_dt, 1e-12)
        timing = max(0.0, 1.0 - timing_error / max(self.dt_tolerance_fraction, 1e-12))
        jump = 1.0 if eef_delta_norm <= self.max_position_jump else 0.0
        velocity = 1.0 if joint_speed_norm <= self.max_joint_speed else 0.0
        return float(min(timing, jump, velocity))


@dataclass
class SequentialRiskAccumulator:
    """Causal evidence accumulator with hysteresis and explicit unknown handling."""
    alarm_threshold: float
    reset_threshold: float
    drift: float
    decay: float = 0.95
    evidence_clip: float = 3.0
    unknown_patience: int = 3
    risk: float = 0.0
    alarm: bool = False
    unknown_run: int = 0

    def __post_init__(self):
        if self.reset_threshold >= self.alarm_threshold:
            raise ValueError("reset_threshold must be below alarm_threshold")
        if not 0 <= self.decay <= 1 or self.unknown_patience < 1:
            raise ValueError("invalid sequential parameters")

    def update(self, lr: float, reliability: float, decision: Decision):
        # Unknown is not converted to negative evidence. It is separately exposed
        # so the controller can slow, observe, or request replanning.
        if decision == "unknown" or not np.isfinite(lr):
            self.unknown_run += 1
            self.risk *= self.decay
        else:
            self.unknown_run = 0
            transformed = float(np.clip(lr, -self.evidence_clip, self.evidence_clip))
            evidence = float(np.clip(reliability, 0.0, 1.0)) * transformed
            self.risk = max(0.0, self.decay * self.risk + evidence - self.drift)
        if self.risk >= self.alarm_threshold:
            self.alarm = True
        elif self.risk <= self.reset_threshold:
            self.alarm = False
        return {
            "risk": self.risk,
            "alarm": self.alarm,
            "unknown_run": self.unknown_run,
            "protective_observation": self.unknown_run >= self.unknown_patience,
        }

    def reset(self):
        self.risk = 0.0
        self.alarm = False
        self.unknown_run = 0
