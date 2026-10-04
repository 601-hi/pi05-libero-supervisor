#!/usr/bin/env python3
"""Compare paired normal/disturbed wrist motion across development suites."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


HORIZONS = (1, 3, 5)


def measurements(features: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "directional_grid": np.linalg.norm(features[..., :32], axis=-1),
        "magnitude_grid": np.linalg.norm(features[..., 32:48], axis=-1),
        "center_magnitude": features[..., 48],
        "ring_magnitude": features[..., 52],
        "radial_energy": np.linalg.norm(features[..., 56:64], axis=-1),
        "tangential_energy": np.linalg.norm(features[..., 64:72], axis=-1),
        "center_to_ring": features[..., 48] / (features[..., 52] + 1e-9),
    }


def summarize(values: np.ndarray) -> dict[str, float]:
    return {
        "median": float(np.median(values)),
        "p25": float(np.percentile(values, 25)),
        "p75": float(np.percentile(values, 75)),
    }


def temporal_measurements(one_step_features: np.ndarray, horizon: int) -> dict[str, np.ndarray]:
    count = len(one_step_features)
    path_directional = np.full(count, np.nan)
    net_directional = np.full(count, np.nan)
    path_magnitude = np.full(count, np.nan)
    for end in range(horizon - 1, count):
        window = one_step_features[end - horizon + 1:end + 1]
        path_directional[end] = np.linalg.norm(window[:, :32], axis=1).sum()
        net_directional[end] = np.linalg.norm(window[:, :32].sum(0))
        path_magnitude[end] = np.linalg.norm(window[:, 32:48], axis=1).sum()
    return {
        "step_path_directional": path_directional,
        "step_net_directional": net_directional,
        "step_path_magnitude": path_magnitude,
    }


def analyze_pair(name: str, normal_path: Path, disturbed_path: Path) -> dict:
    normal = np.load(normal_path, allow_pickle=False)
    disturbed = np.load(disturbed_path, allow_pickle=False)
    common = np.intersect1d(normal["action_indices"], disturbed["action_indices"])
    normal_lookup = {int(v): i for i, v in enumerate(normal["action_indices"])}
    disturbed_lookup = {int(v): i for i, v in enumerate(disturbed["action_indices"])}
    ni = np.asarray([normal_lookup[int(v)] for v in common])
    di = np.asarray([disturbed_lookup[int(v)] for v in common])
    active = disturbed["disturbance_active"][di]
    active_positions = np.flatnonzero(active)
    if not len(active_positions):
        raise ValueError(f"{name} has no disturbed steps")
    onset = int(active_positions[0])
    end = int(active_positions[-1])
    normal_values = measurements(normal["features"][ni])
    disturbed_values = measurements(disturbed["features"][di])
    phases = {
        "pre5": np.arange(max(0, onset - 5), onset),
        "same_chunk5": np.arange(onset, min(onset + 5, end + 1)),
        "replanned5": np.arange(min(onset + 5, end + 1), end + 1),
    }
    result = {"name": name, "onset_action": int(common[onset]), "end_action": int(common[end]), "horizons": {}}
    for horizon_index, horizon in enumerate(HORIZONS):
        normal_temporal = temporal_measurements(normal["features"][ni, 0], horizon)
        disturbed_temporal = temporal_measurements(disturbed["features"][di, 0], horizon)
        horizon_result = {}
        for phase, positions in phases.items():
            valid = normal["valid"][ni[positions], horizon_index] & disturbed["valid"][di[positions], horizon_index]
            selected = positions[valid]
            metric_result = {}
            for metric in normal_values:
                n = normal_values[metric][selected, horizon_index]
                d = disturbed_values[metric][selected, horizon_index]
                ratio = d / (n + 1e-9)
                metric_result[metric] = summarize(ratio)
            for metric in normal_temporal:
                n = normal_temporal[metric][selected]
                d = disturbed_temporal[metric][selected]
                finite = np.isfinite(n) & np.isfinite(d)
                metric_result[metric] = summarize(d[finite] / (n[finite] + 1e-9))
            horizon_result[phase] = {"count": int(len(selected)), "disturbed_to_normal": metric_result}
        result["horizons"][str(horizon)] = horizon_result
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", action="append", nargs=3, metavar=("NAME", "NORMAL", "DISTURBED"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    pairs = [analyze_pair(name, Path(normal), Path(disturbed)) for name, normal, disturbed in args.pair]
    report = {
        "calibration_status": "paired development pilot; not a detector evaluation",
        "pair_count": len(pairs),
        "alignment": "action_t -> wrist_frame_t to wrist_frame_t_plus_1",
        "pairs": pairs,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
