"""T_A_B maps column vectors from frame B into frame A; distances are meters."""
import cv2
import numpy as np


def transform(rotation, translation):
    t = np.eye(4)
    t[:3, :3] = rotation
    t[:3, 3] = np.asarray(translation).reshape(3)
    return t


def rotation_deg(rotation):
    return float(np.degrees(np.linalg.norm(cv2.Rodrigues(rotation)[0])))


def solve(robot, target, method='PARK', min_samples=12, min_span_deg=15.0):
    robot, target = np.asarray(robot), np.asarray(target)
    if len(robot) != len(target) or len(robot) < max(3, min_samples):
        raise ValueError(f'Need at least {max(3, min_samples)} paired samples')
    for poses in (robot, target):
        if poses.shape != (len(robot), 4, 4) or not np.isfinite(poses).all():
            raise ValueError('Samples must be finite 4x4 matrices')
    # Rotations about parallel axes do not constrain a full hand-eye transform.
    relative = np.array([cv2.Rodrigues(robot[0, :3, :3].T @ p[:3, :3])[0].ravel()
                         for p in robot[1:]])
    singular = np.linalg.svd(relative, compute_uv=False)
    if len(singular) < 2 or singular[1] < np.radians(min_span_deg):
        raise ValueError('Insufficient rotation diversity: rotate about at least two axes')
    allowed = ('TSAI', 'PARK', 'HORAUD', 'ANDREFF', 'DANIILIDIS')
    if method not in allowed:
        raise ValueError(f'Method must be one of {allowed}')
    rot, pos = cv2.calibrateHandEye(list(robot[:, :3, :3]), list(robot[:, :3, 3]),
                                  list(target[:, :3, :3]), list(target[:, :3, 3]),
                                  method=getattr(cv2, 'CALIB_HAND_EYE_' + method))
    result = transform(rot, pos)
    if (not np.isfinite(result).all() or
            not np.allclose(rot.T @ rot, np.eye(3), atol=1e-5) or
            not np.isclose(np.linalg.det(rot), 1.0, atol=1e-5)):
        raise ValueError('Hand-eye solver returned an invalid transform')
    boards = robot @ result @ target
    center = boards[:, :3, 3].mean(axis=0)
    u, _, vt = np.linalg.svd(boards[:, :3, :3].mean(axis=0))
    mean_rot = u @ np.diag([1., 1., np.linalg.det(u @ vt)]) @ vt
    metrics = {
        'board_translation_rms_m': float(np.sqrt(np.mean(np.sum(
            (boards[:, :3, 3] - center)**2, axis=1)))),
        'board_rotation_rms_deg': float(np.sqrt(np.mean([
            rotation_deg(mean_rot.T @ b[:3, :3])**2 for b in boards])))}
    return result, metrics
