from dataclasses import replace

import pytest

from vla_supervisor.geometry_measurement import GeometryProjectionResult
from vla_supervisor.goal_measurement import ValidatedImagePoint
from vla_supervisor.goal_progress import GoalProgressConfig, GoalProgressMonitor
from vla_supervisor.goal_progress_adapter import (
    GoalProgressAdapterInput, build_approach_observation,
)


DOMAIN="camera:tool"
TRUST={"geometry-cal-v1","target-review-v1"}


def point(xy, role, target_id, validation_id):
    return ValidatedImagePoint(xy,.01,10,DOMAIN,role,target_id,validation_id)


def inputs(**changes):
    gripper=GeometryProjectionResult(point((.2,.2),'fingertip_midpoint','',
        'geometry-cal-v1'),(20,20),'validated_geometry_projection')
    values=dict(action_index=10,phase='approach',target_id='bowl-a',eef_position=(0,0,1),
        measurement_domain=DOMAIN,gripper=gripper,
        target=point((.5,.6),'grasp_reference','bowl-a','target-review-v1'),
        phase_reliable=True,target_identity_reliable=True,fixed_camera_geometry=True)
    values.update(changes)
    return GoalProgressAdapterInput(**values)


def test_calibrated_same_frame_pair_builds_distance_observation():
    obs=build_approach_observation(inputs(),trusted_validation_ids=TRUST)
    assert obs.goal_error == pytest.approx(.5) and obs.error_bound == pytest.approx(.02)
    assert obs.observable and obs.identity_reliable and obs.domain_verified
    assert obs.geometry_compensated


def test_uncalibrated_gripper_projection_fails_closed():
    uncalibrated=GeometryProjectionResult(None,(20,20),'tool_projection_uncalibrated')
    obs=build_approach_observation(inputs(gripper=uncalibrated),trusted_validation_ids=TRUST)
    assert obs.goal_error is None and not obs.observable and not obs.domain_verified


def test_wrong_target_identity_is_not_silently_used():
    wrong=point((.5,.6),'grasp_reference','plate-b','target-review-v1')
    obs=build_approach_observation(inputs(target=wrong),trusted_validation_ids=TRUST)
    assert not obs.identity_reliable and obs.target_id == ''


def test_time_or_domain_mismatch_fails_closed():
    target=replace(inputs().target,action_index=11)
    obs=build_approach_observation(inputs(target=target),trusted_validation_ids=TRUST)
    assert obs.goal_error is None and not obs.observable
    target=replace(inputs().target,domain='another-camera')
    obs=build_approach_observation(inputs(target=target),trusted_validation_ids=TRUST)
    assert obs.goal_error is None and not obs.observable


def test_moving_camera_distance_is_rejected_by_monitor():
    obs=build_approach_observation(inputs(fixed_camera_geometry=False),trusted_validation_ids=TRUST)
    config=GoalProgressConfig('cal',DOMAIN,'image_distance',4,2,5,.01,.05,.1,.02,.05)
    assert GoalProgressMonitor(config).observe(obs).state == 'unknown'
