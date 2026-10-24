import numpy as np


def matrix(value, name='transform'):
    t = np.asarray(value, dtype=float)
    if t.shape != (4, 4) or not np.isfinite(t).all():
        raise ValueError(f'{name} must be a finite 4x4 transform')
    if (not np.allclose(t[3], [0, 0, 0, 1]) or
            not np.allclose(t[:3, :3].T @ t[:3, :3], np.eye(3), atol=1e-5) or
            not np.isclose(np.linalg.det(t[:3, :3]), 1, atol=1e-5)):
        raise ValueError(f'{name} is not a rigid transform')
    return t


def base_detection(observation, tcp_color, detection):
    camera = matrix(observation) @ matrix(tcp_color)
    center = (camera @ np.r_[detection.point_camera_m, 1.])[:3]
    axis = camera[:3, :3] @ np.asarray(detection.long_axis_camera)
    if np.linalg.norm(axis[:2]) < .5:
        raise ValueError('Brick long axis is not approximately horizontal')
    return center, float(np.arctan2(axis[1], axis[0]))


def pick_pose(center, yaw, reference, reference_yaw_deg, grip_depth):
    pose = matrix(reference).copy()
    # A brick long axis is undirected: take the shortest yaw modulo pi.
    delta = (yaw - np.radians(reference_yaw_deg) + np.pi/2) % np.pi - np.pi/2
    c, s = np.cos(delta), np.sin(delta)
    pose[:3, :3] = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ pose[:3, :3]
    pose[:3, 3] = center
    pose[2, 3] -= grip_depth
    return pose


def above(pose, z):
    result = matrix(pose).copy()
    if z < result[2, 3]:
        raise ValueError('Transit height is below a target')
    result[2, 3] = z
    return result
