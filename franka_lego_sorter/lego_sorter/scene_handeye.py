"""Offline hand-eye candidates from measured robot poses and a static RGB-D scene.

T_A_B maps B coordinates into A. A reference camera snapshot defines the fixed
scene frame. Each registration supplies T_camera_i_reference. This module does
not move hardware and does not declare physical picking accuracy established.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import cv2
import numpy as np
import yaml
from .camera import Frame
from .geometry import matrix
from .scene_registration import register_rgbd


class CalibrationRejected(ValueError):
    def __init__(self, reason, diagnostics=None):
        super().__init__(reason)
        self.diagnostics = dict(diagnostics or {}, accepted=False, reason=str(reason),
                                physical_commissioning_verified=False)


DEFAULTS = dict(min_samples=12, min_rotation_span_deg=15.,
                min_max_rotation_deg=12., max_pair_rotation_disagreement_deg=1.5,
                max_scene_translation_rms_m=.003, max_scene_translation_m=.006,
                max_scene_rotation_rms_deg=.8, max_scene_rotation_deg=1.5,
                max_leaveout_translation_m=.006, max_leaveout_rotation_deg=1.5,
                max_solution_translation_spread_m=.004,
                max_solution_rotation_spread_deg=.8,
                max_mount_translation_m=.35, method='PARK')


def _core():
    """Reuse the adjacent implemented calibration core rather than duplicate it."""
    path = (Path(__file__).resolve().parents[2] / 'franka_realsense_handeye_ros2' /
            'franka_realsense_handeye' / 'core.py')
    if not path.is_file():
        raise CalibrationRejected(f'Existing calibration core not found: {path}')
    spec = importlib.util.spec_from_file_location('_existing_franka_handeye_core', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _angle(rotation):
    return float(np.degrees(np.linalg.norm(cv2.Rodrigues(rotation)[0])))


def _mean(poses):
    poses = np.asarray(poses)
    u, _, vt = np.linalg.svd(poses[:, :3, :3].mean(axis=0))
    rotation = u @ np.diag([1., 1., np.linalg.det(u @ vt)]) @ vt
    result = np.eye(4)
    result[:3, :3] = rotation
    result[:3, 3] = poses[:, :3, 3].mean(axis=0)
    return result


def _pose_errors(poses, reference):
    return (np.linalg.norm(poses[:, :3, 3]-reference[:3, 3], axis=1),
            np.array([_angle(reference[:3, :3].T @ p[:3, :3]) for p in poses]))


def solve_scene_handeye(robot_poses, target_poses, config=None):
    """Return a numerically validated candidate, or reject with diagnostic data.

    Leave-one-out validation excludes the evaluated robot/scene pose from the
    fitted hand-eye transform. It still shares the static RGB-D reference frame,
    so a physical point/grasp validation remains required after this function.
    """
    c = dict(DEFAULTS, **(config or {}))
    diagnostics = dict(sample_count=len(robot_poses), parameters=c,
                       physical_commissioning_verified=False)
    def reject(reason):
        raise CalibrationRejected(reason, diagnostics)
    if len(robot_poses) != len(target_poses):
        reject('Robot and registered scene observations are not paired')
    if len(robot_poses) < max(4, int(c['min_samples'])):
        reject(f"Need at least {max(4, int(c['min_samples']))} measured paired observations")
    try:
        robot = np.array([matrix(p, 'T_base_tcp') for p in robot_poses])
        target = np.array([matrix(p, 'T_camera_reference') for p in target_poses])
    except ValueError as exc:
        reject(str(exc))
    relative = np.array([cv2.Rodrigues(robot[0, :3, :3].T @ p[:3, :3])[0].ravel()
                         for p in robot[1:]])
    singular = np.linalg.svd(relative, compute_uv=False)
    max_angle = float(np.degrees(np.max(np.linalg.norm(relative, axis=1))))
    diagnostics['rotation_excitation'] = dict(singular_values_deg=np.degrees(singular).tolist(),
                                              max_rotation_from_reference_deg=max_angle)
    if singular[1] < np.radians(c['min_rotation_span_deg']) or max_angle < c['min_max_rotation_deg']:
        reject('Insufficient independent multi-axis robot rotation; translations do not determine the camera lever arm')
    disagreements = []
    for i in range(len(robot)):
        for j in range(i):
            # Relative rotations are conjugate under a rigid wrist mount and
            # must have equal angles even before the unknown transform is solved.
            a = _angle(robot[j, :3, :3].T @ robot[i, :3, :3])
            b = _angle(target[i, :3, :3] @ target[j, :3, :3].T)
            disagreements.append(abs(a-b))
    diagnostics['max_pair_rotation_disagreement_deg'] = max(disagreements)
    if max(disagreements) > c['max_pair_rotation_disagreement_deg']:
        reject('Robot/visual relative rotations disagree; scene moved or registration/frame pairing is inconsistent')
    try:
        core = _core()
        transform, _ = core.solve(robot, target, method=c['method'],
                                  min_samples=c['min_samples'], min_span_deg=c['min_rotation_span_deg'])
    except (ValueError, cv2.error) as exc:
        reject('Existing hand-eye solver rejected data: '+str(exc))
    if np.linalg.norm(transform[:3, 3]) > c['max_mount_translation_m']:
        reject('Solved camera offset exceeds the configured wrist-mount envelope')
    fixed = robot @ transform @ target
    fixed_mean = _mean(fixed)
    translations, rotations = _pose_errors(fixed, fixed_mean)
    metrics = dict(translation_rms_m=float(np.sqrt(np.mean(translations**2))),
                   translation_max_m=float(translations.max()),
                   rotation_rms_deg=float(np.sqrt(np.mean(rotations**2))),
                   rotation_max_deg=float(rotations.max()))
    diagnostics['training_scene_residuals'] = metrics
    if (metrics['translation_rms_m'] > c['max_scene_translation_rms_m'] or
            metrics['translation_max_m'] > c['max_scene_translation_m'] or
            metrics['rotation_rms_deg'] > c['max_scene_rotation_rms_deg'] or
            metrics['rotation_max_deg'] > c['max_scene_rotation_deg']):
        reject('Reconstructed static scene is inconsistent across measured poses')
    heldout = []
    fitted = []
    for excluded in range(len(robot)):
        keep = np.arange(len(robot)) != excluded
        try:
            leave_x, _ = core.solve(robot[keep], target[keep], method=c['method'],
                                    min_samples=3, min_span_deg=c['min_rotation_span_deg'])
        except (ValueError, cv2.error) as exc:
            reject(f'Leave-one-out fit {excluded} is unobservable or invalid: {exc}')
        train_scene = _mean(robot[keep] @ leave_x @ target[keep])
        predicted = robot[excluded] @ leave_x @ target[excluded]
        dt, dr = _pose_errors(np.array([predicted]), train_scene)
        heldout.append(dict(excluded_index=excluded, translation_m=float(dt[0]), rotation_deg=float(dr[0])))
        fitted.append(leave_x)
    diagnostics['leave_one_out'] = heldout
    fitted = np.array(fitted)
    spread_t, spread_r = _pose_errors(fitted, transform)
    diagnostics['solution_stability'] = dict(max_translation_m=float(spread_t.max()),
                                             max_rotation_deg=float(spread_r.max()))
    if (max(r['translation_m'] for r in heldout) > c['max_leaveout_translation_m'] or
            max(r['rotation_deg'] for r in heldout) > c['max_leaveout_rotation_deg']):
        reject('Held-out scene observations fail the independent residual bound')
    if (spread_t.max() > c['max_solution_translation_spread_m'] or
            spread_r.max() > c['max_solution_rotation_spread_deg']):
        reject('Hand-eye transform is unstable when one observation is removed')
    diagnostics.update(accepted=True, status='validated_candidate',
                       reason='Numerical gates passed; independent physical point/transfer verification still required')
    return dict(T_tcp_color=transform.tolist(), T_color_tcp=np.linalg.inv(transform).tolist(),
                T_base_reference=fixed_mean.tolist(), diagnostics=diagnostics,
                convention='T_A_B maps B column coordinates into A; meters')


def _load_observation(path):
    with np.load(path, allow_pickle=False) as data:
        if 'T_base_tcp' not in data:
            raise CalibrationRejected(f'{path}: no measured T_base_tcp')
        metadata = json.loads(str(data['metadata_json']))
        frame = Frame(data['color_bgr'], data['depth_m'], json.loads(str(data['intrinsics_json'])),
                      float(data['captured_monotonic']), metadata)
        pose = matrix(data['T_base_tcp'], 'T_base_tcp')
        for key, maximum in [('robot_drift_m', .001), ('robot_drift_rad', .005)]:
            if key not in data or float(data[key]) > maximum:
                raise CalibrationRejected(f'{path}: missing/excessive stationary-capture {key}')
        return frame, pose


def calibrate_observations(paths, config=None):
    c = dict(DEFAULTS, **(config or {}))
    paths = [Path(p).expanduser().resolve() for p in paths]
    diagnostics = dict(observations=[str(p) for p in paths], sample_count=len(paths))
    if len(set(paths)) != len(paths):
        raise CalibrationRejected('Duplicate observation paths are not independent measurements', diagnostics)
    if len(paths) < max(4, int(c['min_samples'])):
        raise CalibrationRejected(f"Need at least {max(4, int(c['min_samples']))} measured observations; only {len(paths)} supplied", diagnostics)
    try:
        observations = [_load_observation(path) for path in paths]
    except CalibrationRejected as exc:
        exc.diagnostics.update(diagnostics)
        raise
    except (OSError, KeyError, ValueError) as exc:
        raise CalibrationRejected('Cannot load measured observations: '+str(exc), diagnostics) from exc
    reference, _ = observations[0]
    metadata = reference.metadata
    for frame, _ in observations[1:]:
        if frame.metadata != metadata or frame.intrinsics != reference.intrinsics:
            raise CalibrationRejected('Camera identity/settings/intrinsics changed between observations', diagnostics)
    robot = [pose for _, pose in observations]
    target = [np.eye(4)]
    registrations = []
    for path, (frame, _) in zip(paths[1:], observations[1:]):
        try:
            registered = register_rgbd(reference, frame, c.get('registration', {}))
        except ValueError as exc:
            diagnostics['registration_results'] = registrations
            raise CalibrationRejected(f'{path}: visual registration rejected: {exc}', diagnostics) from exc
        target.append(registered.T_current_reference)
        registrations.append(dict(path=str(path), metrics=registered.metrics))
    try:
        result = solve_scene_handeye(robot, target, c)
    except CalibrationRejected as exc:
        exc.diagnostics.update(diagnostics, registration_results=registrations)
        raise
    result['diagnostics'].update(diagnostics, registration_results=registrations)
    result['camera_metadata'] = metadata
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('observations', nargs='+')
    parser.add_argument('--output', required=True, help='Diagnostics/result artifact, written even on rejection')
    parser.add_argument('--config', help='Optional YAML scene_handeye settings')
    args = parser.parse_args(argv)
    config = {}
    if args.config:
        value = yaml.safe_load(Path(args.config).read_text()) or {}
        config = value.get('scene_handeye', value)
    try:
        result = calibrate_observations(args.observations, config)
        code = 0
    except CalibrationRejected as exc:
        result = dict(diagnostics=exc.diagnostics)
        code = 2
    path = Path(args.output).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(yaml.safe_dump(result, sort_keys=False))
    temp.replace(path)
    print(json.dumps(result.get('diagnostics', {}), indent=2))
    return code


if __name__ == '__main__':
    sys.exit(main())
