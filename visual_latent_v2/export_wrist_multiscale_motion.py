#!/usr/bin/env python3
"""Export causally aligned multi-scale wrist optical-flow measurements.

This module deliberately exports measurements rather than anomaly labels or a
thresholded detector.  It uses only camera images and observable robot state.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


HORIZONS = (1, 3, 5)


def grid_mean(array: np.ndarray, grid: int = 4) -> np.ndarray:
    height, width = array.shape[:2]
    return np.asarray([
        array[y * height // grid:(y + 1) * height // grid,
              x * width // grid:(x + 1) * width // grid].mean()
        for y in range(grid) for x in range(grid)
    ], dtype=np.float32)


def region_stats(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    selected = values[mask]
    if not len(selected):
        return np.zeros(4, dtype=np.float32)
    return np.asarray([
        selected.mean(), selected.std(), np.percentile(selected, 75), np.percentile(selected, 95)
    ], dtype=np.float32)


def flow_feature(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    before_gray = cv2.cvtColor(before, cv2.COLOR_RGB2GRAY)
    after_gray = cv2.cvtColor(after, cv2.COLOR_RGB2GRAY)
    flow = cv2.calcOpticalFlowFarneback(before_gray, after_gray, None, .5, 3, 15, 3, 5, 1.2, 0)
    magnitude = np.linalg.norm(flow, axis=2)
    height, width = magnitude.shape
    yy, xx = np.mgrid[:height, :width]
    dx = xx - (width - 1) / 2
    dy = yy - (height - 1) / 2
    radius = np.sqrt(dx * dx + dy * dy)
    ux, uy = dx / (radius + 1e-6), dy / (radius + 1e-6)
    radial = flow[..., 0] * ux + flow[..., 1] * uy
    tangential = -flow[..., 0] * uy + flow[..., 1] * ux
    center = radius <= .28 * min(height, width)
    ring = (radius > .28 * min(height, width)) & (radius <= .48 * min(height, width))
    return np.r_[
        grid_mean(flow[..., 0]), grid_mean(flow[..., 1]), grid_mean(magnitude),
        region_stats(magnitude, center), region_stats(magnitude, ring),
        region_stats(radial, center), region_stats(radial, ring),
        region_stats(tangential, center), region_stats(tangential, ring),
    ].astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--trace', type=Path, required=True)
    parser.add_argument('--visual', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()

    steps = [json.loads(line) for line in args.trace.open(encoding='utf-8') if line.strip()]
    steps = [row for row in steps if row.get('event') == 'step']
    steps.sort(key=lambda row: int(row['action_index']))
    raw = np.load(args.visual, allow_pickle=False)
    indices = raw['action_indices'].astype(int)
    if indices.tolist() != [int(row['action_index']) for row in steps]:
        raise ValueError('visual action_indices do not match trace steps')
    wrist = raw['wrist_images']
    if len(wrist) != len(steps):
        raise ValueError('wrist frame count does not match trace steps')

    feature_dim = len(flow_feature(wrist[0], wrist[0]))
    features = np.zeros((len(steps), len(HORIZONS), feature_dim), dtype=np.float32)
    valid = np.zeros((len(steps), len(HORIZONS)), dtype=bool)
    cumulative_intended_translation = np.zeros((len(steps), len(HORIZONS), 3), dtype=np.float32)
    cumulative_executed_translation = np.zeros_like(cumulative_intended_translation)
    for end in range(len(steps) - 1):
        for h_index, horizon in enumerate(HORIZONS):
            start = end - horizon + 1
            if start < 0:
                continue
            features[end, h_index] = flow_feature(wrist[start], wrist[end + 1])
            intended = np.asarray([steps[j]['intended_target_translation'] for j in range(start, end + 1)])
            executed = np.asarray([steps[j]['executed_target_translation'] for j in range(start, end + 1)])
            cumulative_intended_translation[end, h_index] = intended.sum(0)
            cumulative_executed_translation[end, h_index] = executed.sum(0)
            valid[end, h_index] = True

    output = {
        'features': features,
        'valid': valid,
        'action_indices': indices,
        'cumulative_intended_translation': cumulative_intended_translation,
        'cumulative_executed_translation': cumulative_executed_translation,
        'disturbance_active': np.asarray([row.get('disturbance_active', False) for row in steps], dtype=bool),
        'metadata_json': np.asarray(json.dumps({
            'camera': 'wrist', 'horizons': HORIZONS, 'feature_dim_per_horizon': feature_dim,
            'alignment': 'action_end -> wrist_frame_(end-horizon+1) to wrist_frame_(end+1)',
            'feature_groups': {'grid_flow_x': 16, 'grid_flow_y': 16, 'grid_magnitude': 16,
                               'center_ring_magnitude_radial_tangential_stats': 24},
            'uses_task_id': False, 'uses_future_beyond_action_result': False,
        }, ensure_ascii=False)),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **output)
    print(json.dumps({'rows': len(steps), 'horizons': HORIZONS, 'feature_dim': feature_dim,
                      'valid_by_horizon': valid.sum(0).tolist()}, ensure_ascii=False))


if __name__ == '__main__':
    main()
