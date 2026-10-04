"""Runtime EEF-to-image projection with explicit calibration uncertainty."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ProjectedGripper:
    center_xy: tuple[float, float]
    uncertainty_p90_px: float
    calibration_model: str


@dataclass(frozen=True)
class ProjectionDomain:
    camera_id: str
    image_shape: tuple[int, int]
    preprocessing_fingerprint: str


@dataclass(frozen=True)
class ProjectionDomainCheck:
    valid: bool
    status: str


@dataclass(frozen=True)
class DomainBoundProjection:
    projection: ProjectedGripper | None
    domain_check: ProjectionDomainCheck


class GripperPixelProjector:
    """Project robot-reported EEF positions using an empirical RGB calibration."""

    def __init__(self, mean, std, transform, *, model: str, uncertainty_p90_px: float,
                 domain: ProjectionDomain | None = None) -> None:
        if model not in {"affine", "quadratic"}:
            raise ValueError(f"unsupported projection model: {model}")
        self.mean = np.asarray(mean, dtype=float)
        self.std = np.asarray(std, dtype=float)
        self.transform = np.asarray(transform, dtype=float)
        if self.mean.shape != (3,) or self.std.shape != (3,) or np.any(self.std <= 0):
            raise ValueError("mean/std must be valid length-three vectors")
        expected_rows = 4 if model == "affine" else 10
        if self.transform.shape != (expected_rows, 2):
            raise ValueError(f"transform must have shape ({expected_rows}, 2)")
        if uncertainty_p90_px < 0:
            raise ValueError("uncertainty must be non-negative")
        self.model = model
        self.uncertainty_p90_px = float(uncertainty_p90_px)
        self.domain = domain

    @classmethod
    def from_fit_file(cls, path: str | Path) -> "GripperPixelProjector":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        normalization = payload["coordinate_normalization"]
        best = payload["best_model"]
        domain_payload = payload.get("projection_domain")
        domain = None
        if domain_payload is not None:
            shape = tuple(int(value) for value in domain_payload["image_shape"])
            if len(shape) != 2 or any(value <= 0 for value in shape):
                raise ValueError("projection domain image_shape must contain two positive integers")
            domain = ProjectionDomain(
                camera_id=str(domain_payload["camera_id"]),
                image_shape=shape,
                preprocessing_fingerprint=str(domain_payload["preprocessing_fingerprint"]),
            )
        return cls(
            normalization["mean"], normalization["std"], best["transform"],
            model=best["model"], uncertainty_p90_px=best["loo_p90_error_px"],
            domain=domain,
        )

    def validate_domain(
        self, *, camera_id: str, image_shape: tuple[int, int], preprocessing_fingerprint: str
    ) -> ProjectionDomainCheck:
        """Require an exact acquisition-domain match before safety use.

        Legacy calibration files remain loadable for reproducibility, but are
        deliberately not considered valid online calibrations.
        """
        if self.domain is None:
            return ProjectionDomainCheck(False, "legacy_unbound")
        if str(camera_id) != self.domain.camera_id:
            return ProjectionDomainCheck(False, "camera_id_mismatch")
        if tuple(int(value) for value in image_shape) != self.domain.image_shape:
            return ProjectionDomainCheck(False, "image_shape_mismatch")
        if str(preprocessing_fingerprint) != self.domain.preprocessing_fingerprint:
            return ProjectionDomainCheck(False, "preprocessing_mismatch")
        return ProjectionDomainCheck(True, "matched")

    def _features(self, xyz: np.ndarray) -> np.ndarray:
        normalized = (xyz - self.mean) / self.std
        x, y, z = normalized
        if self.model == "affine":
            return np.asarray([1.0, x, y, z])
        return np.asarray([1.0, x, y, z, x*x, y*y, z*z, x*y, x*z, y*z])

    def project(self, eef_position_xyz) -> ProjectedGripper:
        xyz = np.asarray(eef_position_xyz, dtype=float)
        if xyz.shape != (3,) or not np.all(np.isfinite(xyz)):
            raise ValueError("eef_position_xyz must be a finite length-three vector")
        pixel = self._features(xyz) @ self.transform
        return ProjectedGripper(
            (float(pixel[0]), float(pixel[1])), self.uncertainty_p90_px, self.model
        )

    def project_for_domain(
        self, eef_position_xyz, *, camera_id: str, image_shape: tuple[int, int],
        preprocessing_fingerprint: str,
    ) -> DomainBoundProjection:
        """Project only when acquisition metadata exactly matches calibration."""
        check = self.validate_domain(
            camera_id=camera_id,
            image_shape=image_shape,
            preprocessing_fingerprint=preprocessing_fingerprint,
        )
        if not check.valid:
            return DomainBoundProjection(None, check)
        return DomainBoundProjection(self.project(eef_position_xyz), check)
