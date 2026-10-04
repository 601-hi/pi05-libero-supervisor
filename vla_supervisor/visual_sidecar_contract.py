"""Validate public-sensor geometry metadata before visual supervision."""
from dataclasses import dataclass
from typing import Mapping
import numpy as np


@dataclass(frozen=True)
class SidecarContractResult:
    valid: bool
    errors: tuple[str,...]
    frames: int


def validate_geometry_sidecar(data: Mapping[str,object]):
    required=('action_indices','agent_images','wrist_images','agent_camera_intrinsic',
      'wrist_camera_intrinsic','agent_camera_to_world','wrist_camera_to_world',
      'eef_positions','eef_quaternions','gripper_qpos','camera_names','image_transform','saved_image_shape')
    required=required+('geometry_schema_version','observation_time','eef_reference',
      'eef_quaternion_convention','camera_extrinsic_convention',
      'projection_flip_x','projection_flip_y')
    errors=[]
    missing=[k for k in required if k not in data]
    if missing:return SidecarContractResult(False,tuple('missing:'+k for k in missing),0)
    action=np.asarray(data['action_indices']);n=len(action)
    if action.ndim!=1 or (n>1 and np.any(np.diff(action)<=0)):errors.append('action_indices_not_strict')
    for key in ('agent_images','wrist_images','agent_camera_to_world','wrist_camera_to_world',
                'eef_positions','eef_quaternions','gripper_qpos'):
        if len(np.asarray(data[key]))!=n:errors.append(key+':length_mismatch')
    if np.asarray(data['agent_camera_to_world']).shape!=(n,4,4):errors.append('agent_pose_shape')
    if np.asarray(data['wrist_camera_to_world']).shape!=(n,4,4):errors.append('wrist_pose_shape')
    for key in ('agent_camera_intrinsic','wrist_camera_intrinsic'):
        matrix=np.asarray(data[key],float)
        if matrix.shape!=(3,3) or not np.isfinite(matrix).all():errors.append(key+':invalid')
    shape=np.asarray(data['saved_image_shape']).tolist()
    if len(shape)!=2 or any(int(v)<=0 for v in shape):errors.append('saved_image_shape:invalid')
    elif n:
        for key in ('agent_images','wrist_images'):
            if list(np.asarray(data[key]).shape[1:3]) != [int(shape[0]),int(shape[1])]:
                errors.append(key+':shape_domain_mismatch')
    if str(np.asarray(data['image_transform']).item())!='rotate180_then_resize_with_pad':
        errors.append('unknown_image_transform')
    expected_metadata={
      'geometry_schema_version':3,'observation_time':'pre_action',
      'eef_reference':'robot0_grip_site','eef_quaternion_convention':'xyzw',
      'camera_extrinsic_convention':'opencv_camera_to_world'}
    for key, expected in expected_metadata.items():
        value=np.asarray(data[key]).item()
        if value != expected: errors.append(key+':mismatch')
    if bool(np.asarray(data['projection_flip_x']).item()) is not True:
        errors.append('projection_flip_x:mismatch')
    if bool(np.asarray(data['projection_flip_y']).item()) is not False:
        errors.append('projection_flip_y:mismatch')
    for key in ('agent_camera_to_world','wrist_camera_to_world'):
        matrix=np.asarray(data[key],float)
        if not np.isfinite(matrix).all(): errors.append(key+':nonfinite')
        elif n and np.any(np.abs(np.linalg.det(matrix[:,:3,:3])-1)>1e-5):
            errors.append(key+':not_proper_rotation')
    quaternions=np.asarray(data['eef_quaternions'],float)
    if not np.isfinite(quaternions).all(): errors.append('eef_quaternions:nonfinite')
    elif n and np.any(np.abs(np.linalg.norm(quaternions,axis=1)-1)>1e-4):
        errors.append('eef_quaternions:not_unit')
    names=[str(x) for x in np.asarray(data['camera_names']).tolist()]
    if names!=['agentview','robot0_eye_in_hand']:errors.append('camera_names_mismatch')
    return SidecarContractResult(not errors,tuple(errors),n)
