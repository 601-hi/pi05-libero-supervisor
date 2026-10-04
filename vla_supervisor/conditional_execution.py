"""Stateful deployment adapter for the frozen conditional dual-expert ensemble.

The adapter reproduces the source-frozen score without using task identity,
reward, success, disturbance labels, or object state.
"""
from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any, Mapping, Sequence
import json

import numpy as np


LOW = np.asarray([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -.0175, -2.8973], np.float32)
HIGH = np.asarray([2.8973, 1.7628, 2.8973, -.0698, 2.8973, 3.7525, 2.8973], np.float32)
REACH_M = .855


class ConditionalDualExpertScorer:
    """Callable scorer compatible with ``CallbackExecutionMonitor``."""

    def __init__(self, model_paths: Sequence[str | Path], *, window: int = 10,
                 beta: float = 1.0, reliability_calibration: str | Path | None = None):
        if len(model_paths) < 3:
            raise ValueError("at least three ensemble members are required")
        import torch
        from torch import nn
        from cross_suite_generalization.conditional_relational_dual_expert import make_model

        torch.set_num_threads(1)
        self.torch = torch
        self.members = []
        for path in model_paths:
            try:
                checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
            except TypeError:
                # LIBERO's Python 3.8 environment carries an older PyTorch
                # without the weights_only keyword. Checkpoints are trusted
                # local project artifacts, not user-provided files.
                checkpoint = torch.load(Path(path), map_location="cpu")
            config = checkpoint["config"]
            model = make_model(torch, nn, config["input_dim"], config["output_dim"], config["seed"])
            model.load_state_dict(checkpoint["state_dict"])
            model.eval()
            self.members.append((checkpoint, model))
        self.window = int(window)
        self.beta = float(beta)
        self.support_best_logp_min = None
        self.support_ensemble_std_max = None
        if reliability_calibration is not None:
            calibration = json.loads(Path(reliability_calibration).read_text(encoding="utf-8"))
            if not calibration.get("source_only"):
                raise ValueError("reliability calibration must declare source_only")
            self.support_best_logp_min = float(np.percentile(calibration["best_logp_reference"], 1))
            self.support_ensemble_std_max = float(np.percentile(calibration["ensemble_std_reference"], 99))
        self.reset()

    def reset(self) -> None:
        self.initial_eef = None
        self.previous_response = np.zeros(3, np.float32)
        self.previous_progress = 0.0
        self.robust_scores: deque[float] = deque(maxlen=self.window)

    @staticmethod
    def _gripper_fraction(state: Mapping[str, Any]) -> float:
        q = np.asarray(state["gripper_qpos"], dtype=np.float32)
        return float(np.clip(abs(q[0] - q[1]) / .08, 0, 1))

    def __call__(self, intended_action, state_before, state_after, history):
        action = np.asarray(intended_action, np.float32)
        previous_actions = [np.asarray(item["intended_action"], np.float32) for item in history[-5:]][::-1]
        lag_actions = [action, *previous_actions]
        if len(lag_actions) < 6:
            # Match the offline adapter: its first exported row is action 5,
            # whose episode-relative state is the feature origin.
            return 0.0, 0.0, {"status": "warming_up_command_history",
                                        "history_fill": len(lag_actions), "history_required": 6}
        q = np.asarray(state_before["joint_pos"], np.float32)
        eef0 = np.asarray(state_before["eef_pos"], np.float32)
        eef1 = np.asarray(state_after["eef_pos"], np.float32)
        if self.initial_eef is None:
            self.initial_eef = eef0.copy()

        velocity = np.r_[.05 * np.clip(action[:3], -1, 1),
                         .5 * np.clip(action[3:6], -1, 1)] * 20.0
        current = velocity / 20.0
        translation = current[:3]
        norm = float(np.linalg.norm(translation))
        unit = translation / (norm + 1e-8)
        qnorm = 2 * (q - LOW) / (HIGH - LOW) - 1
        margin = float(np.min(np.minimum((q - LOW) / (HIGH - LOW), (HIGH - q) / (HIGH - LOW))))
        relative = (eef0 - self.initial_eef) / REACH_M
        rotation_norm = float(np.linalg.norm(current[3:6]) / np.pi)
        x = np.r_[qnorm, margin, relative, self._gripper_fraction(state_before), unit,
                  norm / REACH_M, rotation_norm, self.previous_response / REACH_M,
                  self.previous_progress].astype(np.float32)[None]

        response = eef1 - eef0
        lag_commands = np.asarray([.05 * np.clip(item[:3], -1, 1) for item in lag_actions], np.float32)
        lag_norms = np.linalg.norm(lag_commands, axis=1)
        progress_values = np.sum(lag_commands * response[None], axis=1) / (lag_norms * lag_norms + 1e-10)
        progress_values = np.clip(progress_values, -2, 2).astype(np.float32)
        progress = float(progress_values[0])
        y = progress_values[None]
        normal_logp, abnormal_logp = [], []
        with self.torch.no_grad():
            for checkpoint, model in self.members:
                X = self.torch.from_numpy(((x - checkpoint["xcenter"]) / checkpoint["xscale"]).astype(np.float32))
                Y = self.torch.from_numpy(((y - checkpoint["ycenter"]) / checkpoint["yscale"]).astype(np.float32))
                normal_logp.append(float(model.logp(X, Y, False).item()))
                abnormal_logp.append(float(model.logp(X, Y, True).item()))
        likelihood_ratios = np.asarray(abnormal_logp) - np.asarray(normal_logp)
        ensemble_mean = float(likelihood_ratios.mean())
        ensemble_std = float(likelihood_ratios.std())
        robust_step_score = ensemble_mean - self.beta * ensemble_std
        self.robust_scores.append(robust_step_score)
        sequence_score = float(np.mean(self.robust_scores))
        ready = len(self.robust_scores) == self.window
        best_logp = max(float(np.mean(normal_logp)), float(np.mean(abnormal_logp)))
        supported = (self.support_best_logp_min is None or
                     (best_logp >= self.support_best_logp_min and
                      ensemble_std <= self.support_ensemble_std_max))
        self.previous_response = response.astype(np.float32)
        self.previous_progress = progress
        evidence = {
            "normal_logp_mean": float(np.mean(normal_logp)),
            "abnormal_logp_mean": float(np.mean(abnormal_logp)),
            "likelihood_ratio_mean": ensemble_mean,
            "ensemble_std": ensemble_std,
            "robust_step_score": robust_step_score,
            "sequence_score": sequence_score,
            "sequence_window_fill": len(self.robust_scores),
            "sequence_window": self.window,
            "source_only_calibration": True,
            "target_domain_guarantee": False,
            "support_best_logp": best_logp,
            "support_best_logp_min": self.support_best_logp_min,
            "support_ensemble_std_max": self.support_ensemble_std_max,
            "support_status": "supported" if supported else "unknown_low_support",
        }
        # A partial window must never alarm. Confidence expresses availability,
        # not the posterior probability of failure.
        confidence = 1.0 if ready and supported else 0.0
        return (sequence_score if ready else float("-inf")), confidence, evidence


class CalibratedFourStateConditionalScorer:
    """Convert the frozen conditional ensemble into deployment four states."""

    def __init__(self, model_paths: Sequence[str | Path], *, calibration_path: str | Path,
                 window: int = 10, beta: float = 1.0, expected_dt: float = 0.05):
        from cross_suite_generalization.reliability_sequence_supervisor import (
            ReliabilityCalibration, SensorQualityMonitor,
        )
        self.base = ConditionalDualExpertScorer(model_paths, window=window, beta=beta)
        self.calibration = ReliabilityCalibration.from_json(calibration_path)
        self.sensor = SensorQualityMonitor(expected_dt=expected_dt)
        self.expected_dt = float(expected_dt)

    def reset(self) -> None:
        self.base.reset()

    def __call__(self, intended_action, state_before, state_after, history):
        _, _, evidence = self.base(intended_action, state_before, state_after, history)
        required = {"normal_logp_mean", "abnormal_logp_mean", "ensemble_std"}
        sequence_ready = (
            evidence.get("sequence_window_fill") == evidence.get("sequence_window")
            if "sequence_window" in evidence else False
        )
        if not required.issubset(evidence) or not sequence_ready:
            return {
                **evidence,
                "decision": "unknown",
                "lr": float("nan"),
                "reliability": 0.0,
                "sensor_quality": 0.0,
                "four_state_status": (
                    "accumulating_sequence" if required.issubset(evidence) else "warming_up"
                ),
            }
        eef_before = np.asarray(state_before["eef_pos"], dtype=float)
        eef_after = np.asarray(state_after["eef_pos"], dtype=float)
        joint_velocity = np.asarray(state_after.get("joint_vel", []), dtype=float)
        finite = bool(
            np.isfinite(eef_before).all() and np.isfinite(eef_after).all()
            and np.isfinite(joint_velocity).all()
        )
        sensor_quality = self.sensor.score(
            self.expected_dt,
            float(np.linalg.norm(eef_after - eef_before)),
            float(np.linalg.norm(joint_velocity)),
            finite=finite,
        )
        calibrated = self.calibration.score(
            float(evidence["normal_logp_mean"]),
            float(evidence["abnormal_logp_mean"]),
            float(evidence["ensemble_std"]),
            sensor_quality,
        )
        return {
            **evidence,
            **calibrated,
            "sensor_quality": sensor_quality,
            "four_state_status": "ready",
        }
