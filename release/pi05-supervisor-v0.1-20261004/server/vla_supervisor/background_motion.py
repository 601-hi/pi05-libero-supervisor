"""Robust 2-D background motion estimation without depth or simulator state."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class BackgroundMotionEstimate:
    homography: np.ndarray | None
    valid: bool
    tracked_points: int
    inlier_ratio: float
    median_reprojection_error_px: float
    spatial_coverage: float
    confidence: float


@dataclass(frozen=True)
class ResidualMotion:
    median_px: float
    p90_px: float
    coherent_fraction: float
    valid_pixels: int


def _gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image.astype(np.uint8, copy=False)
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image.astype(np.uint8, copy=False), cv2.COLOR_RGB2GRAY)
    raise ValueError("image must be HxW grayscale or HxWx3 RGB")


def _coverage(points_xy: np.ndarray, shape: tuple[int, int], grid: int = 4) -> float:
    if not len(points_xy):
        return 0.0
    height, width = shape
    x_cells = np.clip((points_xy[:, 0] * grid / width).astype(int), 0, grid - 1)
    y_cells = np.clip((points_xy[:, 1] * grid / height).astype(int), 0, grid - 1)
    return len(set(zip(x_cells.tolist(), y_cells.tolist()))) / float(grid * grid)


def estimate_background_motion(
    previous_rgb: np.ndarray,
    current_rgb: np.ndarray,
    exclusion_mask: np.ndarray | None = None,
    *,
    max_corners: int = 500,
    ransac_threshold_px: float = 2.0,
    minimum_points: int = 20,
    minimum_inlier_ratio: float = 0.55,
    minimum_coverage: float = 0.25,
) -> BackgroundMotionEstimate:
    """Estimate a majority homography using forward-backward checked sparse flow."""
    previous = _gray(previous_rgb)
    current = _gray(current_rgb)
    if previous.shape != current.shape:
        raise ValueError("frame shapes must match")
    feature_mask = np.full(previous.shape, 255, dtype=np.uint8)
    if exclusion_mask is not None:
        if exclusion_mask.shape != previous.shape:
            raise ValueError("exclusion mask shape must match image")
        feature_mask[np.asarray(exclusion_mask, dtype=bool)] = 0
    points0 = cv2.goodFeaturesToTrack(
        previous, maxCorners=max_corners, qualityLevel=0.01,
        minDistance=5, blockSize=5, mask=feature_mask,
    )
    if points0 is None or len(points0) < minimum_points:
        return BackgroundMotionEstimate(None, False, 0 if points0 is None else len(points0), 0, np.inf, 0, 0)
    points1, status1, _ = cv2.calcOpticalFlowPyrLK(previous, current, points0, None)
    points0_back, status0, _ = cv2.calcOpticalFlowPyrLK(current, previous, points1, None)
    if points1 is None or points0_back is None:
        return BackgroundMotionEstimate(None, False, 0, 0, np.inf, 0, 0)
    fb_error = np.linalg.norm(points0_back[:, 0] - points0[:, 0], axis=1)
    valid_flow = status1[:, 0].astype(bool) & status0[:, 0].astype(bool) & (fb_error < 1.5)
    source = points0[valid_flow, 0]
    target = points1[valid_flow, 0]
    if len(source) < minimum_points:
        return BackgroundMotionEstimate(None, False, len(source), 0, np.inf, _coverage(source, previous.shape), 0)
    homography, inliers = cv2.findHomography(source, target, cv2.RANSAC, ransac_threshold_px)
    if homography is None or inliers is None:
        return BackgroundMotionEstimate(None, False, len(source), 0, np.inf, _coverage(source, previous.shape), 0)
    inlier_mask = inliers[:, 0].astype(bool)
    projected = cv2.perspectiveTransform(source[:, None, :], homography)[:, 0]
    errors = np.linalg.norm(projected - target, axis=1)
    ratio = float(inlier_mask.mean())
    median_error = float(np.median(errors[inlier_mask])) if inlier_mask.any() else np.inf
    coverage = _coverage(source[inlier_mask], previous.shape)
    error_quality = max(0.0, 1.0 - median_error / max(ransac_threshold_px, 1e-6))
    confidence = float(np.clip(ratio * np.sqrt(coverage) * error_quality, 0.0, 1.0))
    valid = bool(
        len(source) >= minimum_points
        and ratio >= minimum_inlier_ratio
        and coverage >= minimum_coverage
        and np.isfinite(median_error)
    )
    return BackgroundMotionEstimate(homography, valid, len(source), ratio, median_error, coverage, confidence)


def residual_motion_in_mask(
    previous_rgb: np.ndarray,
    current_rgb: np.ndarray,
    mask: np.ndarray,
    estimate: BackgroundMotionEstimate,
    *,
    residual_threshold_px: float = 1.5,
) -> ResidualMotion:
    """Compare dense observed flow in a mask with the fitted static-background flow."""
    if not estimate.valid or estimate.homography is None:
        return ResidualMotion(np.nan, np.nan, np.nan, 0)
    previous = _gray(previous_rgb)
    current = _gray(current_rgb)
    selected = np.asarray(mask, dtype=bool)
    if selected.shape != previous.shape:
        raise ValueError("mask shape must match image")
    flow = cv2.calcOpticalFlowFarneback(previous, current, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    yy, xx = np.nonzero(selected)
    if not len(xx):
        return ResidualMotion(np.nan, np.nan, np.nan, 0)
    source = np.column_stack((xx, yy)).astype(np.float32)
    predicted = cv2.perspectiveTransform(source[:, None, :], estimate.homography)[:, 0]
    predicted_flow = predicted - source
    observed_flow = flow[yy, xx]
    norm = np.linalg.norm(observed_flow - predicted_flow, axis=1)
    return ResidualMotion(
        float(np.median(norm)),
        float(np.percentile(norm, 90)),
        float(np.mean(norm > residual_threshold_px)),
        int(len(norm)),
    )
