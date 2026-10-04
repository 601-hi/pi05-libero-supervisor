import unittest
import numpy as np
from conditional_relational_dual_expert import conditional_features

class FeatureTests(unittest.TestCase):
 def test_shape_finite_and_no_current_response_leak(self):
  n=5;data={'command_history_cartesian_velocity':np.ones((n,6,6),np.float32),'state_joint_position':np.tile(np.array([0,0,0,-1.5,0,1.5,0],np.float32),(n,1)),'state_eef_pose_xyz_euler':np.zeros((n,6),np.float32),'state_gripper_position':np.zeros((n,1),np.float32),'response_eef_delta_xyz_euler':np.zeros((n,6),np.float32)}
  ep=np.zeros(n,np.int32);x1,y1=conditional_features(data,ep,15);data2={k:v.copy() for k,v in data.items()};data2['response_eef_delta_xyz_euler'][-1,:3]+=.001;x2,y2=conditional_features(data2,ep,15)
  self.assertEqual(x1.shape,(n,21));self.assertEqual(y1.shape,(n,6));self.assertTrue(np.isfinite(x1).all());self.assertTrue(np.array_equal(x1[-1],x2[-1]));self.assertFalse(np.array_equal(y1[-1],y2[-1]))
 def test_episode_history_resets(self):
  n=4;data={'command_history_cartesian_velocity':np.ones((n,6,6),np.float32),'state_joint_position':np.tile(np.array([0,0,0,-1.5,0,1.5,0],np.float32),(n,1)),'state_eef_pose_xyz_euler':np.zeros((n,6),np.float32),'state_gripper_position':np.zeros((n,1),np.float32),'response_eef_delta_xyz_euler':np.ones((n,6),np.float32)}
  x,_=conditional_features(data,np.array([0,0,1,1]),15);self.assertTrue(np.allclose(x[0,-4:],0));self.assertTrue(np.allclose(x[2,-4:],0))
if __name__=='__main__':unittest.main()
