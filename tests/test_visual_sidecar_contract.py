import unittest
from copy import deepcopy
import numpy as np
from vla_supervisor.visual_sidecar_contract import validate_geometry_sidecar


def sample(n=2):
    return {'action_indices':np.arange(n),'agent_images':np.zeros((n,4,4,3),np.uint8),
      'wrist_images':np.zeros((n,4,4,3),np.uint8),'agent_camera_intrinsic':np.eye(3),
      'wrist_camera_intrinsic':np.eye(3),'agent_camera_to_world':np.tile(np.eye(4),(n,1,1)),
      'wrist_camera_to_world':np.tile(np.eye(4),(n,1,1)),'eef_positions':np.zeros((n,3)),
      'eef_quaternions':np.tile(np.array([0,0,0,1.]),(n,1)),'gripper_qpos':np.zeros((n,2)),
      'camera_names':np.array(['agentview','robot0_eye_in_hand']),
      'image_transform':np.array('rotate180_then_resize_with_pad'),'saved_image_shape':np.array([4,4]),
      'geometry_schema_version':np.array(3),'observation_time':np.array('pre_action'),
      'eef_reference':np.array('robot0_grip_site'),'eef_quaternion_convention':np.array('xyzw'),
      'camera_extrinsic_convention':np.array('opencv_camera_to_world'),
      'projection_flip_x':np.array(True),'projection_flip_y':np.array(False)}


class TestSidecarContract(unittest.TestCase):
    def test_valid(self): self.assertTrue(validate_geometry_sidecar(sample()).valid)
    def test_missing(self):
        x=sample();del x['eef_quaternions'];self.assertIn('missing:eef_quaternions',validate_geometry_sidecar(x).errors)
    def test_alignment(self):
        x=sample();x['eef_positions']=np.zeros((1,3));self.assertFalse(validate_geometry_sidecar(x).valid)
    def test_pose_shape(self):
        x=sample();x['wrist_camera_to_world']=np.zeros((2,3,4));self.assertIn('wrist_pose_shape',validate_geometry_sidecar(x).errors)
    def test_image_domain(self):
        x=sample();x['saved_image_shape']=np.array([5,4]);self.assertFalse(validate_geometry_sidecar(x).valid)
    def test_transform(self):
        x=sample();x['image_transform']=np.array('mystery');self.assertIn('unknown_image_transform',validate_geometry_sidecar(x).errors)
    def test_semantic_metadata(self):
        x=sample();x['observation_time']=np.array('post_action');self.assertIn('observation_time:mismatch',validate_geometry_sidecar(x).errors)
    def test_rotation_and_quaternion_validity(self):
        x=sample();x['agent_camera_to_world'][0,:3,:3]=0;self.assertIn('agent_camera_to_world:not_proper_rotation',validate_geometry_sidecar(x).errors)
        x=sample();x['eef_quaternions'][0]=0;self.assertIn('eef_quaternions:not_unit',validate_geometry_sidecar(x).errors)
    def test_projection_axis_convention(self):
        x=sample();x['projection_flip_y']=np.array(True);self.assertIn('projection_flip_y:mismatch',validate_geometry_sidecar(x).errors)

if __name__=='__main__':unittest.main()
