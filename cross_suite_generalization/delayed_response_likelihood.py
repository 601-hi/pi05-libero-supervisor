"""Lag-marginalized linear response baseline for causal dynamics auditing.

This is a transparent baseline and data-quality gate, not the final nonlinear
dynamics expert.  All fitting must use normal training data only.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class LagComponent:
    lag_steps: int
    gain: float
    residual_variance: float
    weight: float
    correlation: float


def _pair(command: np.ndarray, response: np.ndarray, lag: int) -> tuple[np.ndarray, np.ndarray]:
    """Pair command t with response t+lag; positive lag means delayed response."""
    start = max(0, -lag)
    end = min(len(command), len(response) - lag)
    return command[start:end], response[start + lag:end + lag]


def _pooled_pairs(
    episodes: Sequence[tuple[np.ndarray, np.ndarray]], lag: int
) -> tuple[np.ndarray, np.ndarray]:
    pairs = [_pair(np.asarray(command), np.asarray(response), lag) for command, response in episodes]
    pairs = [(x, y) for x, y in pairs if len(x)]
    if not pairs:
        raise ValueError(f"no command/response pairs for lag {lag}")
    return np.concatenate([x for x, _ in pairs]), np.concatenate([y for _, y in pairs])


def fit_lag_bank(
    episodes: Sequence[tuple[np.ndarray, np.ndarray]],
    candidate_lags: Sequence[int],
    *,
    temperature: float = 1.0,
    variance_floor: float = 1e-8,
) -> tuple[LagComponent, ...]:
    """Fit scalar-gain Gaussian components and validation-style soft weights."""
    if not episodes or not candidate_lags:
        raise ValueError("episodes and candidate_lags must be non-empty")
    if any(int(lag) < 0 for lag in candidate_lags):
        raise ValueError("deployable lag components must be causal (non-negative)")
    if temperature <= 0 or variance_floor <= 0:
        raise ValueError("temperature and variance_floor must be positive")
    raw = []
    for lag in candidate_lags:
        command, response = _pooled_pairs(episodes, int(lag))
        x, y = command.reshape(-1), response.reshape(-1)
        gain = float((x @ y) / (x @ x + 1e-12))
        residual = y - gain * x
        variance = float(max(np.mean(residual ** 2), variance_floor))
        correlation = float(np.corrcoef(x, y)[0, 1])
        raw.append((int(lag), gain, variance, correlation))
    # Lower normal residual variance receives more mass, without hard-selecting a lag.
    logits = -np.log([item[2] for item in raw]) / temperature
    weights = np.exp(logits - np.max(logits))
    weights /= weights.sum()
    return tuple(
        LagComponent(lag, gain, variance, float(weight), correlation)
        for (lag, gain, variance, correlation), weight in zip(raw, weights)
    )


def lag_marginal_nll(
    command_history: np.ndarray,
    measured_response: np.ndarray,
    components: Sequence[LagComponent],
) -> float:
    """Return -log sum_l w_l N(response; gain_l * command[t-l], var_l)."""
    history = np.asarray(command_history, dtype=float)
    response = np.asarray(measured_response, dtype=float).reshape(-1)
    if history.ndim != 2 or history.shape[1] != response.size:
        raise ValueError("command_history must be [time, dimension] and match response")
    log_terms = []
    for component in components:
        if component.lag_steps < 0 or component.lag_steps >= len(history):
            continue
        command = history[-1 - component.lag_steps].reshape(-1)
        residual = response - component.gain * command
        dimension = response.size
        log_density = -0.5 * (
            dimension * np.log(2 * np.pi * component.residual_variance)
            + float(residual @ residual) / component.residual_variance
        )
        log_terms.append(np.log(component.weight) + log_density)
    if not log_terms:
        raise ValueError("command history does not cover any lag component")
    maximum = max(log_terms)
    return float(-(maximum + np.log(np.exp(np.asarray(log_terms) - maximum).sum())))


def episode_alignment_quality(
    command: np.ndarray,
    response: np.ndarray,
    candidate_lags: Sequence[int],
    *,
    minimum_correlation: float = 0.3,
    minimum_response_to_command_rms: float = 0.1,
) -> dict:
    """Flag streams that are excited but cannot support command/response identification."""
    command = np.asarray(command, dtype=float)
    response = np.asarray(response, dtype=float)
    command_rms = float(np.sqrt(np.mean(command ** 2)))
    response_rms = float(np.sqrt(np.mean(response ** 2)))
    correlations = {}
    for lag in candidate_lags:
        x, y = _pair(command, response, int(lag))
        correlations[int(lag)] = float(np.corrcoef(x.reshape(-1), y.reshape(-1))[0, 1])
    best_lag = max(correlations, key=correlations.get)
    ratio = response_rms / (command_rms + 1e-12)
    accepted = correlations[best_lag] >= minimum_correlation and ratio >= minimum_response_to_command_rms
    return {
        "accepted_for_identification": bool(accepted),
        "command_rms": command_rms,
        "response_rms": response_rms,
        "response_to_command_rms": ratio,
        "best_lag_steps": int(best_lag),
        "best_correlation": correlations[best_lag],
        "correlation_by_lag": correlations,
    }
