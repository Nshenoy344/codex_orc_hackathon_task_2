"""Conservative markerless registration of a *fixed* textured RGB-D scene.

T_current_reference maps a reference optical-camera point into the current
optical-camera frame. This is not a hand-eye transform. A moved or rearranged
pallet invalidates its use as a calibration target even if registration fits.
"""
from dataclasses import dataclass
import math
import cv2
import numpy as np


class RegistrationError(ValueError):
    pass


@dataclass
class RegistrationResult:
    T_current_reference: np.ndarray
    metrics: dict
    matches_reference_px: np.ndarray | None = None
    matches_current_px: np.ndarray | None = None
    inlier_mask: np.ndarray | None = None

    def to_dict(self):
        return dict(T_current_reference=self.T_current_reference.tolist(), metrics=self.metrics,
                    matches_reference_px=None if self.matches_reference_px is None else self.matches_reference_px.tolist(),
                    matches_current_px=None if self.matches_current_px is None else self.matches_current_px.tolist(),
                    inlier_mask=None if self.inlier_mask is None else self.inlier_mask.tolist())


def _rigid_fit(source, target):
    a, b = source.mean(axis=0), target.mean(axis=0)
    ac, bc = source-a, target-b
    if np.linalg.matrix_rank(ac, tol=1e-8) < 2 or np.linalg.matrix_rank(bc, tol=1e-8) < 2:
        raise RegistrationError('Correspondences are collinear or repeated')
    u, _, vt = np.linalg.svd(ac.T@bc)
    rotation = vt.T@np.diag([1., 1., np.linalg.det(vt.T@u.T)])@u.T
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = b-rotation@a
    return transform


def _errors(transform, source, target):
    predicted = source@transform[:3, :3].T+transform[:3, 3]
    return np.linalg.norm(predicted-target, axis=1)


def _spread_3d(points, config):
    centered = points-points.mean(axis=0)
    singular = np.linalg.svd(centered, compute_uv=False)/np.sqrt(len(points))
    if singular[1] < float(config.get('min_second_axis_std_m', .008)):
        raise RegistrationError(f'Insufficient noncollinear 3D spread: principal std {singular.tolist()} m')
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    projected = centered@vt.T
    spans = np.percentile(projected, 95, axis=0)-np.percentile(projected, 5, axis=0)
    if spans[1] < float(config.get('min_second_axis_span_m', .025)):
        raise RegistrationError(f'Inliers occupy too small a scene region: principal spans {spans.tolist()} m')
    return dict(principal_std_m=singular.tolist(), principal_span_m=spans.tolist(),
                rank=int(np.linalg.matrix_rank(centered, tol=1e-5)))


def estimate_rigid_ransac(reference_xyz, current_xyz, config=None):
    """RANSAC+Kabsch with deterministic seed, residual and metric-spread gates."""
    config = config or {}
    source, target = np.asarray(reference_xyz, float), np.asarray(current_xyz, float)
    required = max(20, int(config.get('min_inliers', 20)))
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise RegistrationError('Paired correspondence arrays must have shape Nx3')
    if len(source) < required or not np.isfinite(source).all() or not np.isfinite(target).all():
        raise RegistrationError(f'Need at least {required} finite paired 3D correspondences')
    threshold = float(config.get('max_residual_m', .003))
    if threshold <= 0:
        raise RegistrationError('max_residual_m must be positive')
    rng = np.random.default_rng(int(config.get('ransac_seed', 37)))
    maximum = int(config.get('ransac_iterations', 2500))
    best_mask, best_score = None, (-1, -math.inf)
    iterations = 0
    adaptive_limit = maximum
    for iteration in range(maximum):
        iterations = iteration+1
        sample = rng.choice(len(source), 3, replace=False)
        try:
            hypothesis = _rigid_fit(source[sample], target[sample])
        except RegistrationError:
            continue
        errors = _errors(hypothesis, source, target)
        mask = errors <= threshold
        number = int(mask.sum())
        score = (number, -float(np.median(errors[mask])) if number else -math.inf)
        if score > best_score:
            best_mask, best_score = mask, score
            fraction = number/len(source)
            if fraction > 0:
                failure = max(1e-12, 1-fraction**3)
                adaptive_limit = min(adaptive_limit, max(50, int(np.ceil(np.log(.0001)/np.log(failure)))))
        if iterations >= adaptive_limit:
            break
    if best_mask is None or best_mask.sum() < required:
        raise RegistrationError(f'RANSAC found fewer than {required} consistent 3D correspondences')
    mask = best_mask
    for _ in range(5):
        transform = _rigid_fit(source[mask], target[mask])
        errors = _errors(transform, source, target)
        refined = errors <= threshold
        if refined.sum() < required:
            raise RegistrationError('Refined registration lost required inlier support')
        if np.array_equal(refined, mask):
            break
        mask = refined
    transform = _rigid_fit(source[mask], target[mask])
    errors = _errors(transform, source, target)
    # Final quality is computed only from points within the final fitted threshold.
    mask = errors <= threshold
    if mask.sum() < required:
        raise RegistrationError('Final fit has insufficient inliers')
    competing_support = 0
    remaining = np.flatnonzero(~mask)
    if len(remaining) >= required and config.get('check_competing_models', True):
        # Repeated stud patterns can generate two coherent translations. A
        # similarly supported distinct rigid fit is ambiguity, not confidence.
        for _ in range(int(config.get('competing_model_iterations', 500))):
            sample = rng.choice(remaining, 3, replace=False)
            try:
                alternative = _rigid_fit(source[sample], target[sample])
            except RegistrationError:
                continue
            alternative_mask = _errors(alternative, source, target) <= threshold
            support = int(alternative_mask.sum())
            if support < required:
                continue
            translation_delta = np.linalg.norm(alternative[:3, 3]-transform[:3, 3])
            cosine = np.clip((np.trace(alternative[:3, :3].T@transform[:3, :3])-1)/2, -1, 1)
            rotation_delta = np.arccos(cosine)
            if translation_delta > float(config.get('distinct_model_translation_m', .005)) or rotation_delta > float(config.get('distinct_model_rotation_rad', .035)):
                competing_support = max(competing_support, support)
                if support >= max(required, int(mask.sum()*float(config.get('max_competing_support_fraction', .75)))):
                    raise RegistrationError(f'Ambiguous scene: distinct rigid model has {support} matches versus {int(mask.sum())} best inliers')
    fraction = float(mask.mean())
    if fraction < float(config.get('min_inlier_fraction', .30)):
        raise RegistrationError(f'Insufficient inlier fraction: {fraction:.3f}')
    reference_spread = _spread_3d(source[mask], config)
    current_spread = _spread_3d(target[mask], config)
    residuals = errors[mask]
    metrics = dict(correspondence_count=len(source), inlier_count=int(mask.sum()), inlier_fraction=fraction,
                   residual_rms_m=float(np.sqrt(np.mean(residuals**2))),
                   residual_median_m=float(np.median(residuals)), residual_max_m=float(residuals.max()),
                   residual_threshold_m=threshold, ransac_iterations=iterations,
                   competing_model_inlier_count=competing_support,
                   reference_spread=reference_spread, current_spread=current_spread,
                   scene_assumption='The same rigid static scene must be visible; scene motion cannot be distinguished from camera motion')
    return RegistrationResult(transform, metrics, inlier_mask=mask)


def _depth_point(frame, pixel, config):
    x, y = map(float, pixel)
    ix, iy = int(round(x)), int(round(y))
    radius = int(config.get('depth_patch_radius_px', 2))
    h, w = frame.depth_m.shape
    if ix-radius < 0 or iy-radius < 0 or ix+radius >= w or iy+radius >= h:
        return None
    patch = frame.depth_m[iy-radius:iy+radius+1, ix-radius:ix+radius+1]
    valid = np.isfinite(patch) & (patch > float(config.get('min_depth_m', .15))) & (patch < float(config.get('max_depth_m', 2.)))
    if valid.mean() < float(config.get('min_depth_patch_fraction', .70)):
        return None
    values = patch[valid]
    low, high = np.percentile(values, [10, 90])
    if high-low > float(config.get('max_depth_patch_span_m', .004)):
        return None  # skip depth discontinuities and flying pixels
    return frame.deproject(x, y, float(np.median(values)))


def _spread_pixels(pixels, image_shape, config):
    points = np.asarray(pixels, np.float32)
    hull = cv2.convexHull(points.reshape(-1, 1, 2))
    area = float(cv2.contourArea(hull))
    _, _, vt = np.linalg.svd(points-points.mean(axis=0), full_matrices=False)
    principal = (points-points.mean(axis=0))@vt.T
    spans = np.percentile(principal, 95, axis=0)-np.percentile(principal, 5, axis=0)
    if spans[1] < float(config.get('min_second_axis_span_px', 40)):
        raise RegistrationError(f'Insufficient 2D inlier spread: spans {spans.tolist()} px')
    if area < float(config.get('min_inlier_hull_area_px', 1500)):
        raise RegistrationError(f'Inliers concentrate in a small image patch ({area:.1f}px²)')
    # Occupied bins use the inlier bounding box, not full image, so a centered
    # pallet can pass while several repeated studs in one corner cannot.
    minimum, maximum = points.min(axis=0), points.max(axis=0)
    normalized = (points-minimum)/np.maximum(maximum-minimum, 1)
    bins = np.minimum((normalized*4).astype(int), 3)
    occupied = len(set(map(tuple, bins)))
    if occupied < int(config.get('min_occupied_grid_cells', 6)):
        raise RegistrationError(f'Inlier spatial coverage too sparse ({occupied}/16 grid cells)')
    return dict(hull_area_px=area, principal_span_px=spans.tolist(), occupied_grid_cells=occupied,
                image_shape=list(image_shape[:2]))


def register_rgbd(reference, current, config=None):
    """Mutual SIFT ratio matches -> reliable local depths -> rigid RANSAC."""
    config = config or {}
    if not hasattr(cv2, 'SIFT_create'):
        raise RegistrationError('OpenCV SIFT support is required')
    for frame in (reference, current):
        if frame.color_bgr.shape[:2] != frame.depth_m.shape:
            raise RegistrationError('RGB-D registration requires aligned color and depth')
    ref_serial = reference.metadata.get('serial')
    current_serial = current.metadata.get('serial')
    if ref_serial and current_serial and ref_serial != current_serial and not config.get('allow_different_cameras', False):
        raise RegistrationError('Reference/current camera serials differ')
    sift = cv2.SIFT_create(nfeatures=int(config.get('max_keypoints', 4000)),
                           contrastThreshold=float(config.get('sift_contrast_threshold', .025)))
    features = []
    for frame in (reference, current):
        gray = cv2.cvtColor(frame.color_bgr, cv2.COLOR_BGR2GRAY)
        mask = ((frame.depth_m > float(config.get('min_depth_m', .15))) &
                (frame.depth_m < float(config.get('max_depth_m', 2.)))).astype(np.uint8)*255
        if config.get('roi') is not None:
            from .perception import roi_mask
            mask &= roi_mask(frame.color_bgr.shape, config['roi'])
        features.append(sift.detectAndCompute(gray, mask))
    (ref_kp, ref_desc), (cur_kp, cur_desc) = features
    if ref_desc is None or cur_desc is None or min(len(ref_desc), len(cur_desc)) < 20:
        raise RegistrationError('Too few textured RGB-D features for scene registration')
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    ratio = float(config.get('sift_ratio', .70))
    if not 0 < ratio < 1:
        raise RegistrationError('SIFT ratio must lie between zero and one')
    def ratio_matches(a, b):
        return {pair[0].queryIdx: pair[0].trainIdx for pair in matcher.knnMatch(a, b, k=2)
                if len(pair) == 2 and pair[0].distance < ratio*pair[1].distance}
    forward, reverse = ratio_matches(ref_desc, cur_desc), ratio_matches(cur_desc, ref_desc)
    mutual = [(a, b) for a, b in forward.items() if reverse.get(b) == a]
    points_reference, points_current, pixels_reference, pixels_current = [], [], [], []
    used_pairs = set()
    for a, b in mutual:
        rp, cp = ref_kp[a].pt, cur_kp[b].pt
        # SIFT may emit several orientations at the exact same stud. Such copies
        # must not inflate inlier count or spatial support.
        signature = tuple(int(round(value/2)) for value in (*rp, *cp))
        if signature in used_pairs:
            continue
        used_pairs.add(signature)
        r3d, c3d = _depth_point(reference, rp, config), _depth_point(current, cp, config)
        if r3d is None or c3d is None:
            continue
        points_reference.append(r3d)
        points_current.append(c3d)
        pixels_reference.append(rp)
        pixels_current.append(cp)
    if len(points_reference) < max(20, int(config.get('min_inliers', 20))):
        raise RegistrationError(f'Only {len(points_reference)} mutual unambiguous matches have reliable local depth')
    result = estimate_rigid_ransac(points_reference, points_current, config)
    result.matches_reference_px = np.asarray(pixels_reference)
    result.matches_current_px = np.asarray(pixels_current)
    result.metrics.update(reference_keypoints=len(ref_kp), current_keypoints=len(cur_kp),
                          mutual_ratio_matches=len(mutual), reliable_depth_matches=len(points_reference),
                          sift_ratio=ratio,
                          reference_image_spread=_spread_pixels(result.matches_reference_px[result.inlier_mask], reference.color_bgr.shape, config),
                          current_image_spread=_spread_pixels(result.matches_current_px[result.inlier_mask], current.color_bgr.shape, config))
    return result
