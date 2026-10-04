"""Task-agnostic geometry derived from object/target/gripper masks."""
from __future__ import annotations
from dataclasses import asdict,dataclass
import numpy as np

@dataclass(frozen=True)
class MaskRelationGeometry:
    object_area_fraction: float
    target_area_fraction: float
    gripper_area_fraction: float
    object_target_overlap_fraction: float
    object_centroid_in_target_bbox: bool
    gripper_object_boundary_distance_px: float
    image_diagonal_px: float
    def to_dict(self):return asdict(self)

def _mask(value):
    x=np.asarray(value,dtype=bool)
    if x.ndim!=2:raise ValueError(f"mask must be 2D, got {x.shape}")
    return x

def _centroid(mask):
    points=np.argwhere(mask)
    return points.mean(0) if len(points) else None

def _bbox(mask):
    points=np.argwhere(mask)
    if not len(points):return None
    return points.min(0),points.max(0)

def _boundary_distance(left,right):
    a=np.argwhere(left);b=np.argwhere(right)
    if not len(a) or not len(b):return float('inf')
    # Chunked exact distance avoids a large H*W*H*W temporary array.
    best=float('inf')
    for start in range(0,len(a),256):
        delta=a[start:start+256,None,:]-b[None,:,:]
        best=min(best,float(np.sqrt(np.sum(delta*delta,axis=2).min())))
    return best

def relation_geometry(object_mask,target_mask,gripper_mask):
    obj,target,gripper=map(_mask,(object_mask,target_mask,gripper_mask))
    if obj.shape!=target.shape or obj.shape!=gripper.shape:raise ValueError('all masks must share shape')
    pixels=float(obj.size);overlap=float(np.logical_and(obj,target).sum()/max(obj.sum(),1))
    center=_centroid(obj);box=_bbox(target)
    inside=False if center is None or box is None else bool(np.all(center>=box[0]) and np.all(center<=box[1]))
    return MaskRelationGeometry(float(obj.sum()/pixels),float(target.sum()/pixels),
        float(gripper.sum()/pixels),overlap,inside,_boundary_distance(obj,gripper),
        float(np.hypot(*obj.shape)))
