import json

import numpy as np
import pytest

from vla_supervisor.gripper_projection import GripperPixelProjector, ProjectionDomain


def test_affine_projection_and_uncertainty():
    transform = np.asarray([[10, 20], [2, 0], [0, 3], [1, -1]], dtype=float)
    projector = GripperPixelProjector([0, 0, 0], [1, 1, 1], transform, model="affine", uncertainty_p90_px=7.5)
    result = projector.project([1, 2, 3])
    assert result.center_xy == pytest.approx((15, 23))
    assert result.uncertainty_p90_px == 7.5


def test_quadratic_fit_file_round_trip(tmp_path):
    payload = {
        "coordinate_normalization": {"mean": [1, 2, 3], "std": [2, 2, 2]},
        "best_model": {
            "model": "quadratic", "loo_p90_error_px": 12,
            "transform": [[1, 2]] + [[0, 0]] * 9,
        },
    }
    path = tmp_path / "fit.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    result = GripperPixelProjector.from_fit_file(path).project([1, 2, 3])
    assert result.center_xy == (1.0, 2.0)
    assert result.calibration_model == "quadratic"


def test_invalid_eef_position_rejected():
    projector = GripperPixelProjector([0, 0, 0], [1, 1, 1], np.zeros((4, 2)), model="affine", uncertainty_p90_px=0)
    with pytest.raises(ValueError):
        projector.project([1, np.nan, 3])


def test_legacy_fit_is_not_silently_treated_as_domain_matched(tmp_path):
    payload = {
        "coordinate_normalization": {"mean": [0, 0, 0], "std": [1, 1, 1]},
        "best_model": {
            "model": "affine", "loo_p90_error_px": 5,
            "transform": [[0, 0]] * 4,
        },
    }
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    check = GripperPixelProjector.from_fit_file(path).validate_domain(
        camera_id="agentview", image_shape=(224, 224), preprocessing_fingerprint="rotate180-resize224"
    )
    assert not check.valid
    assert check.status == "legacy_unbound"


def test_projection_domain_requires_exact_camera_shape_and_preprocessing():
    projector = GripperPixelProjector(
        [0, 0, 0], [1, 1, 1], np.zeros((4, 2)), model="affine", uncertainty_p90_px=5,
        domain=ProjectionDomain("agentview", (224, 224), "rotate180-resize224"),
    )
    assert projector.validate_domain(
        camera_id="agentview", image_shape=(224, 224), preprocessing_fingerprint="rotate180-resize224"
    ).valid
    assert projector.validate_domain(
        camera_id="agentview", image_shape=(224, 224), preprocessing_fingerprint="raw-resize224"
    ).status == "preprocessing_mismatch"
    assert projector.validate_domain(
        camera_id="wrist", image_shape=(224, 224), preprocessing_fingerprint="rotate180-resize224"
    ).status == "camera_id_mismatch"


def test_domain_bound_projection_abstains_instead_of_returning_wrong_pixel():
    projector = GripperPixelProjector(
        [0, 0, 0], [1, 1, 1], np.zeros((4, 2)), model="affine", uncertainty_p90_px=5,
        domain=ProjectionDomain("agentview", (224, 224), "rotate180-resize224"),
    )
    refused = projector.project_for_domain(
        [0, 0, 0], camera_id="agentview", image_shape=(224, 224),
        preprocessing_fingerprint="raw-resize224",
    )
    assert refused.projection is None
    assert refused.domain_check.status == "preprocessing_mismatch"
    accepted = projector.project_for_domain(
        [0, 0, 0], camera_id="agentview", image_shape=(224, 224),
        preprocessing_fingerprint="rotate180-resize224",
    )
    assert accepted.projection is not None
    assert accepted.domain_check.valid
