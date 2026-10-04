import unittest
import numpy as np
from dataclasses import replace
from vla_supervisor.camera_geometry import (
    CameraCalibration,camera_model_fingerprint,project_world_point,tool_point_world,
)


class TestCameraGeometry(unittest.TestCase):
    def setUp(self):
        self.c=CameraCalibration('cal-v1','fixed',((100,0,50),(0,100,40),(0,0,1)),
            ((1,0,0,0),(0,1,0,0),(0,0,1,0),(0,0,0,1)),80,100)

    def test_projection(self): self.assertEqual(project_world_point((0,0,2),self.c).xy,(50.,40.))
    def test_flip_is_explicit(self):
        self.assertEqual(project_world_point((0,0,2),replace(self.c,flip_x=True,flip_y=True)).xy,(49.,39.))
    def test_outside(self): self.assertFalse(project_world_point((2,0,1),self.c).visible)
    def test_behind(self): self.assertEqual(project_world_point((0,0,-1),self.c).reason,'behind_camera')
    def test_tool_offset_uses_orientation(self):
        rot=np.array([[0,-1,0],[1,0,0],[0,0,1]],float)
        np.testing.assert_allclose(tool_point_world((1,2,3),rot,(.1,0,0)),(1,2.1,3))
    def test_reject_non_rotation(self):
        with self.assertRaises(ValueError): tool_point_world((0,0,0),np.zeros((3,3)),(0,0,0))
    def test_fingerprint_binds_model_not_dynamic_pose(self):
        moved=replace(self.c,calibration_id='other',world_to_camera=(
            (1,0,0,1),(0,1,0,2),(0,0,1,3),(0,0,0,1)))
        flipped=replace(self.c,flip_y=True)
        self.assertEqual(camera_model_fingerprint(self.c),camera_model_fingerprint(moved))
        self.assertNotEqual(camera_model_fingerprint(self.c),camera_model_fingerprint(flipped))

if __name__=='__main__':unittest.main()
