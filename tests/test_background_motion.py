import cv2
import numpy as np

from vla_supervisor.background_motion import estimate_background_motion, residual_motion_in_mask


def textured_frame(seed=7):
    rng = np.random.default_rng(seed)
    image = rng.integers(0, 256, size=(120, 160, 3), dtype=np.uint8)
    return cv2.GaussianBlur(image, (3, 3), 0)


def warp(image, dx, dy):
    matrix = np.asarray([[1, 0, dx], [0, 1, dy], [0, 0, 1]], dtype=np.float32)
    return cv2.warpPerspective(image, matrix, (image.shape[1], image.shape[0])), matrix


def test_recovers_global_background_translation():
    previous = textured_frame()
    current, expected = warp(previous, 3, -2)
    result = estimate_background_motion(previous, current)
    assert result.valid
    point = np.asarray([[[80.0, 60.0]]], dtype=np.float32)
    observed = cv2.perspectiveTransform(point, result.homography)
    target = cv2.perspectiveTransform(point, expected)
    assert np.linalg.norm(observed - target) < 0.5
    assert result.inlier_ratio > 0.9


def test_detects_region_moving_independently_of_background():
    previous = textured_frame()
    current, _ = warp(previous, 2, 0)
    # An independently moving textured patch violates the global background transform.
    patch = previous[40:80, 50:90]
    current[40:80, 58:98] = patch
    mask = np.zeros(previous.shape[:2], dtype=bool)
    mask[42:78, 60:96] = True
    exclusion = np.zeros_like(mask)
    exclusion[35:85, 45:105] = True
    estimate = estimate_background_motion(previous, current, exclusion)
    residual = residual_motion_in_mask(previous, current, mask, estimate)
    assert estimate.valid
    assert residual.median_px > 1.5
    assert residual.p90_px > 4.0
    assert residual.coherent_fraction > 0.5


def test_low_texture_frame_is_rejected():
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    result = estimate_background_motion(frame, frame)
    assert not result.valid
    assert result.homography is None
