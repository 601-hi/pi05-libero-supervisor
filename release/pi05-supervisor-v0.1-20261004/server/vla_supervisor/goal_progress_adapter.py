"""Fail-closed bridge from calibrated image points to goal-progress inputs."""
from __future__ import annotations

from dataclasses import dataclass

from .geometry_measurement import GeometryProjectionResult
from .goal_measurement import ValidatedImagePoint, approach_distance
from .goal_progress import GoalProgressObservation


@dataclass(frozen=True)
class GoalProgressAdapterInput:
    action_index: int
    phase: str
    target_id: str
    eef_position: tuple[float, float, float]
    measurement_domain: str
    gripper: GeometryProjectionResult
    target: ValidatedImagePoint | None
    phase_reliable: bool
    target_identity_reliable: bool
    fixed_camera_geometry: bool


def build_approach_observation(
    value: GoalProgressAdapterInput, *, trusted_validation_ids: set[str]
) -> GoalProgressObservation:
    """Construct an observation without inventing missing visual evidence."""
    distance = approach_distance(
        value.gripper.point,
        value.target,
        trusted_validation_ids=trusted_validation_ids,
        fixed_domain=value.measurement_domain,
    )
    target_matches = bool(
        value.target is not None
        and value.target.target_id
        and value.target.target_id == value.target_id
    )
    trusted = distance.calibrated
    return GoalProgressObservation(
        action_index=value.action_index,
        phase=value.phase,
        target_id=value.target_id if target_matches else "",
        measurement_domain=value.measurement_domain,
        metric_kind="image_distance",
        goal_error=distance.value,
        error_bound=distance.error_bound,
        eef_position=value.eef_position,
        observable=trusted,
        identity_reliable=trusted and value.target_identity_reliable and target_matches,
        phase_reliable=value.phase_reliable,
        domain_verified=trusted,
        geometry_compensated=value.fixed_camera_geometry,
    )
