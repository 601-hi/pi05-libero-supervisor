"""Conservative adapter for independently validated fixed-camera measurements.

This is not a visual detector. Validation identifiers must be supplied by a
trusted calibration/annotation pipeline, never copied from VLM output.
"""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ValidatedImagePoint:
    xy: tuple[float, float]
    error_radius: float
    action_index: int
    domain: str
    semantic_role: str
    target_id: str
    validation_id: str
    visible: bool = True


@dataclass(frozen=True)
class ImageDistance:
    value: float | None
    error_bound: float | None
    reason: str
    calibrated: bool


def approach_distance(gripper, target, *, trusted_validation_ids, fixed_domain):
    """Normalized 2-D approach distance, NOT 3-D clearance or task success.

    Points and radii use a fixed image-diagonal normalization. Error propagation
    follows the triangle inequality, conditional on both input bounds holding.
    A pair of marginal 95% bounds is not automatically a joint 95% bound.
    """
    reject = lambda reason: ImageDistance(None, None, reason, False)
    if gripper is None or target is None:
        return reject('missing_measurement')
    for point in (gripper, target):
        if not point.visible:
            return reject('occluded')
        if not point.validation_id or point.validation_id not in trusted_validation_ids:
            return reject('unvalidated')
        if not fixed_domain or point.domain != fixed_domain:
            return reject('domain_mismatch')
        if len(point.xy) != 2 or not all(math.isfinite(v) and 0 <= v <= 1 for v in point.xy):
            return reject('invalid_coordinates')
        if not math.isfinite(point.error_radius) or point.error_radius < 0:
            return reject('invalid_uncertainty')
    if gripper.action_index != target.action_index:
        return reject('time_mismatch')
    if gripper.semantic_role != 'fingertip_midpoint':
        return reject('not_fingertip_midpoint')
    if target.semantic_role != 'grasp_reference' or not target.target_id:
        return reject('not_validated_grasp_reference')
    return ImageDistance(math.dist(gripper.xy, target.xy),
                         gripper.error_radius + target.error_radius,
                         'validated_2d_approach_only', True)
