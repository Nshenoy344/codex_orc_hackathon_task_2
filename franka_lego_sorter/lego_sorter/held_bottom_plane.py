"""Offline rigid held-bottom geometry, with no nominal LEGO height subtraction.

Planes use upward unit normals and n @ point = offset. A source support plane
becomes an object-bottom plane only if the object rested flat on it when grasped
and did not slip. These functions do not establish either physical condition,
object center, collision clearance, support semantics, or execution approval.
"""
import numpy as np


def _plane(normal, offset):
    n = np.asarray(normal, float)
    length = np.linalg.norm(n)
    if n.shape != (3,) or not np.isfinite(n).all() or length < 1e-9 or not np.isfinite(offset):
        raise ValueError('Finite nonzero plane required')
    return n / length, float(offset) / length


def bottom_plane_in_tcp(T_base_tcp_at_grasp, support_normal_base, support_offset_base):
    """Conditional bottom plane expressed in the actual TCP at grasp closure."""
    T = np.asarray(T_base_tcp_at_grasp, float)
    if T.shape != (4, 4) or not np.isfinite(T).all():
        raise ValueError('Finite 4x4 grasp transform required')
    n, h = _plane(support_normal_base, support_offset_base)
    return T[:3, :3].T @ n, float(h - n @ T[:3, 3])


def release_tcp_z(xy_base, R_base_tcp, bottom_normal_tcp, bottom_offset_tcp,
                  support_normal_base, support_offset_base, footprint_points_tcp,
                  clearance_m):
    """Return TCP Z ensuring every measured bottom footprint point clears plane.

    footprint_points_tcp must be actual bottom-boundary points, not nominal
    body dimensions. Orientation need not match the support. The most downward
    transformed point governs clearance. Only Z is solved; XY and rotation stay
    supplied. A visible-top percentile plane may be used for a loose-placement
    diagnostic, but is not automatically a load-bearing or stud/body plane.
    """
    n, h = _plane(support_normal_base, support_offset_base)
    bn, bh = _plane(bottom_normal_tcp, bottom_offset_tcp)
    R = np.asarray(R_base_tcp, float)
    points = np.asarray(footprint_points_tcp, float)
    xy = np.asarray(xy_base, float)
    if R.shape != (3, 3) or points.ndim != 2 or points.shape[1] != 3 or len(points) < 3 or xy.shape != (2,):
        raise ValueError('Rotation, XY and at least three measured bottom footprint points required')
    if not all(np.isfinite(v).all() for v in (R, points, xy)) or not np.isfinite(clearance_m) or clearance_m < 0:
        raise ValueError('Finite geometry and nonnegative explicit clearance required')
    if n[2] < .5:
        raise ValueError('Support must have an upward normal suitable for a Z-only solve')
    if np.max(np.abs(points @ bn - bh)) > 1e-6:
        raise ValueError('Footprint points must lie on the supplied held bottom plane')
    minimum_offset = float(np.min((points @ R.T) @ n))
    return float((h + clearance_m - n[:2] @ xy - minimum_offset) / n[2])
