#!/usr/bin/env python3
"""CPU runtime for the frozen conditional normal-dynamics expert.

The caller supplies the same deployable pre-step feature vector used during
training (dynamic features + task one-hot) and the measured post-step EEF
translation. This module intentionally has no LIBERO or simulator dependency.
"""
from __future__ import annotations

import collections
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, Optional

import numpy as np
import torch

from dynamics_feature_schema import CONDITION_DIM_WITH_TASK, schema_sha256


class _Expert(torch.nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 128),
            torch.nn.SiLU(),
            torch.nn.Linear(128, 128),
            torch.nn.SiLU(),
            torch.nn.Linear(128, 6),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass(frozen=True)
class RuntimeResult:
    predicted_translation_m: np.ndarray
    predictive_std_m: np.ndarray
    standardized_innovation: np.ndarray
    score: float
    threshold: float
    score_ratio: float
    point_alarm: bool
    persistent_alarm: bool
    observable: bool
    status: str


class NormalDynamicsExpertRuntime:
    """Stateful, causal evaluator for one robot stream."""

    def __init__(self, checkpoint_path: str | Path, profile: str = "90"):
        checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
        if profile not in checkpoint["threshold_sets"]:
            raise ValueError(f"unknown profile {profile!r}; available={sorted(checkpoint['threshold_sets'])}")
        self.input_dim = int(checkpoint["input_dim"])
        if self.input_dim != CONDITION_DIM_WITH_TASK:
            raise ValueError(
                f"checkpoint input_dim={self.input_dim} does not match canonical schema "
                f"dimension={CONDITION_DIM_WITH_TASK}"
            )
        self.feature_schema_sha256 = schema_sha256()
        self.profile = profile
        self.thresholds = {int(k): float(v) for k, v in checkpoint["threshold_sets"][profile].items()}
        self.variance_scale = {
            int(k): np.asarray(v, dtype=np.float64) for k, v in checkpoint["variance_scale"].items()
        }
        self.required = int(checkpoint["persistence"]["required"])
        self.window = int(checkpoint["persistence"]["window"])
        self.observability_threshold_m = float(
            checkpoint.get("observability", {}).get("episode_mean_target_norm_warning_below_m", 0.015)
        )
        self.members = []
        for saved in checkpoint["members"]:
            net = _Expert(self.input_dim)
            net.load_state_dict(saved["state_dict"])
            net.eval()
            self.members.append(
                {
                    "net": net,
                    "x_mean": np.asarray(saved["x_mean"], dtype=np.float64),
                    "x_scale": np.asarray(saved["x_scale"], dtype=np.float64),
                    "y_mean": np.asarray(saved["y_mean"], dtype=np.float64),
                    "y_scale": np.asarray(saved["y_scale"], dtype=np.float64),
                }
            )
        self._point_history: Deque[bool] = collections.deque(maxlen=self.window)
        self._target_norm_history: Deque[float] = collections.deque(maxlen=10)

    def reset(self) -> None:
        self._point_history.clear()
        self._target_norm_history.clear()

    def _predict(self, feature: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        means, variances = [], []
        for member in self.members:
            standardized = (feature - member["x_mean"]) / member["x_scale"]
            tensor = torch.as_tensor(standardized[None, :], dtype=torch.float32)
            with torch.no_grad():
                output = member["net"](tensor).numpy()[0]
            mean = output[:3] * member["y_scale"] + member["y_mean"]
            variance = np.exp(np.clip(output[3:], -6.0, 3.0)) * member["y_scale"] ** 2
            means.append(mean)
            variances.append(variance)
        means = np.asarray(means)
        total_variance = np.mean(variances, axis=0) + np.var(means, axis=0)
        return np.mean(means, axis=0), total_variance

    def update(
        self,
        feature: np.ndarray,
        actual_translation_m: np.ndarray,
        task_id: int,
        intended_target_norm_m: float,
    ) -> RuntimeResult:
        feature = np.asarray(feature, dtype=np.float64)
        actual = np.asarray(actual_translation_m, dtype=np.float64)
        if feature.shape != (self.input_dim,):
            raise ValueError(f"feature must have shape ({self.input_dim},), got {feature.shape}")
        if actual.shape != (3,):
            raise ValueError(f"actual_translation_m must have shape (3,), got {actual.shape}")
        if task_id not in self.thresholds:
            raise ValueError(f"task_id {task_id} has no calibrated threshold")

        mean, variance = self._predict(feature)
        variance = np.maximum(variance * self.variance_scale[task_id], 1e-12)
        innovation = (actual - mean) / np.sqrt(variance)
        score = float(np.dot(innovation, innovation))
        threshold = self.thresholds[task_id]
        point_alarm = score > threshold
        self._point_history.append(point_alarm)
        self._target_norm_history.append(float(intended_target_norm_m))
        persistent_alarm = len(self._point_history) >= self.required and sum(self._point_history) >= self.required
        observable = float(np.mean(self._target_norm_history)) >= self.observability_threshold_m
        if persistent_alarm:
            status = "abnormal"
        elif not observable:
            status = "low_observability"
        else:
            status = "normal_no_alarm"
        return RuntimeResult(
            predicted_translation_m=mean,
            predictive_std_m=np.sqrt(variance),
            standardized_innovation=innovation,
            score=score,
            threshold=threshold,
            score_ratio=score / (threshold + 1e-12),
            point_alarm=point_alarm,
            persistent_alarm=persistent_alarm,
            observable=observable,
            status=status,
        )


def checkpoint_summary(checkpoint_path: str | Path) -> Dict[str, object]:
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    return {
        "input_dim": int(checkpoint["input_dim"]),
        "members": len(checkpoint["members"]),
        "profiles": sorted(checkpoint["threshold_sets"]),
        "persistence": dict(checkpoint["persistence"]),
        "observability": dict(checkpoint.get("observability", {})),
        "runtime_feature_schema_sha256": schema_sha256(),
    }
