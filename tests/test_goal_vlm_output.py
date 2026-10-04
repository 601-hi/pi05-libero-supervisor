import json
import pytest
from vla_supervisor.goal_vlm_output import parse_goal_vlm


def sample(**updates):
    v=dict(gripper_center=[500,500],object_box=[100,200,300,400],destination_box=None,
           phase='unknown',goal_state='unknown',evidence='partly occluded')
    v.update(updates)
    return json.dumps(v)


def test_output_cannot_self_authorize_or_claim_calibration():
    parsed=parse_goal_vlm(sample(calibrated=True,command='stop'))
    assert parsed['valid'] and not parsed['calibrated'] and 'command' not in parsed


@pytest.mark.parametrize('change',[dict(gripper_center=[1200,20]),dict(object_box=[4,3,1,2]),
    dict(gripper_center=[True,2]),dict(phase='failure'),dict(goal_state='probably'),dict(evidence=4)])
def test_invalid_measurements_are_rejected(change):
    assert not parse_goal_vlm(sample(**change))['valid']


def test_missing_unobservable_and_non_json_are_distinct():
    assert parse_goal_vlm(sample(gripper_center=None,object_box=None))['valid']
    assert not parse_goal_vlm('{}')['valid']
    assert not parse_goal_vlm('I think it worked')['valid']
