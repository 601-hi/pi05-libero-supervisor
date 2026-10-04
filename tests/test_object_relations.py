import numpy as np
import pytest
from vla_supervisor.object_relations import relation_geometry

def test_object_inside_target_and_gripper_contact_geometry():
    obj=np.zeros((20,20),bool);obj[8:11,8:11]=1
    target=np.zeros_like(obj);target[5:15,5:15]=1
    gripper=np.zeros_like(obj);gripper[8:11,11:13]=1
    r=relation_geometry(obj,target,gripper)
    assert r.object_centroid_in_target_bbox
    assert r.object_target_overlap_fraction==1.0
    assert r.gripper_object_boundary_distance_px==1.0

def test_missing_object_is_not_fabricated_as_inside():
    empty=np.zeros((10,10),bool);target=empty.copy();target[2:8,2:8]=1
    r=relation_geometry(empty,target,empty)
    assert not r.object_centroid_in_target_bbox
    assert np.isinf(r.gripper_object_boundary_distance_px)

def test_mask_shape_mismatch_rejected():
    with pytest.raises(ValueError):relation_geometry(np.zeros((2,2)),np.zeros((3,3)),np.zeros((2,2)))
