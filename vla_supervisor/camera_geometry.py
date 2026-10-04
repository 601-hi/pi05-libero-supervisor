"""Calibrated world-to-image geometry with explicit image transforms."""
from dataclasses import dataclass
import hashlib
import json
import math
import numpy as np


@dataclass(frozen=True)
class CameraCalibration:
    calibration_id: str
    camera_id: str
    intrinsic: tuple[tuple[float, float, float], ...]
    world_to_camera: tuple[tuple[float, float, float, float], ...]
    image_height: int
    image_width: int
    flip_x: bool = False
    flip_y: bool = False

    def matrices(self):
        if not self.calibration_id or not self.camera_id or self.image_height<1 or self.image_width<1:
            raise ValueError('calibration provenance and image shape are required')
        k=np.asarray(self.intrinsic,float);t=np.asarray(self.world_to_camera,float)
        if k.shape!=(3,3) or t.shape!=(4,4) or not np.isfinite(k).all() or not np.isfinite(t).all():
            raise ValueError('invalid calibration matrices')
        return k,t


@dataclass(frozen=True)
class PixelProjection:
    xy: tuple[float,float] | None
    visible: bool
    reason: str


def camera_model_fingerprint(calibration: CameraCalibration) -> str:
    """Bind measurements to intrinsics and saved-image conventions, not pose."""
    k, _ = calibration.matrices()
    payload = {
        'camera_id': calibration.camera_id,
        'image_height': calibration.image_height,
        'image_width': calibration.image_width,
        'flip_x': calibration.flip_x,
        'flip_y': calibration.flip_y,
        'intrinsic_f64_hex': np.asarray(k, dtype='<f8').tobytes().hex(),
    }
    encoded=json.dumps(payload,sort_keys=True,separators=(',',':')).encode('utf-8')
    return 'camera-model-sha256:'+hashlib.sha256(encoded).hexdigest()


def tool_point_world(eef_position, eef_rotation, local_offset):
    position=np.asarray(eef_position,float);rotation=np.asarray(eef_rotation,float);offset=np.asarray(local_offset,float)
    if (position.shape!=(3,) or rotation.shape!=(3,3) or offset.shape!=(3,)
            or not all(np.isfinite(x).all() for x in (position,rotation,offset))):
        raise ValueError('invalid tool transform')
    if not np.allclose(rotation.T@rotation,np.eye(3),atol=1e-5) or np.linalg.det(rotation)<0.999:
        raise ValueError('eef rotation must be a proper rotation matrix')
    return position+rotation@offset


def project_world_point(point_world, calibration:CameraCalibration):
    k,t=calibration.matrices();p=np.asarray(point_world,float)
    if p.shape!=(3,) or not np.isfinite(p).all(): return PixelProjection(None,False,'invalid_world_point')
    camera=t@np.r_[p,1.]
    if camera[2]<=1e-9: return PixelProjection(None,False,'behind_camera')
    homogeneous=k@camera[:3];x,y=(homogeneous[:2]/homogeneous[2]).tolist()
    if calibration.flip_x:x=calibration.image_width-1-x
    if calibration.flip_y:y=calibration.image_height-1-y
    visible=0<=x<calibration.image_width and 0<=y<calibration.image_height
    return PixelProjection((float(x),float(y)),visible,'visible' if visible else 'outside_image')
