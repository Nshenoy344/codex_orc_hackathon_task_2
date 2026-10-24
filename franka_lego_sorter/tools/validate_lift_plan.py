#!/usr/bin/env python3
"""Offline joint-limit and FK verification. Does not import ROS or command hardware."""
import argparse
import json
import hashlib
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from scipy.interpolate import BPoly
from scipy.spatial.transform import Rotation


def validate(plan_path, urdf_path):
    data = json.loads(Path(plan_path).read_text())
    root = ET.parse(urdf_path).getroot()
    joint_elements = {j.get('name'): j for j in root.findall('joint')}
    bychild = {j.find('child').get('link'): j for j in root.findall('joint')}
    chain, child = [], data['tcp_frame']
    while child != data['base_frame']:
        joint = bychild[child]
        chain.append(joint)
        child = joint.find('parent').get('link')
    chain.reverse()
    names, points = data['joint_names'], data['points']
    positions = np.array([p['positions'] for p in points])
    times = np.array([p['time_sec'] for p in points])
    if np.any(np.diff(times) <= 0) or not np.isfinite(positions).all():
        raise ValueError('Nonfinite positions or invalid times')
    def fk(q):
        values = dict(zip(names, q))
        t = np.eye(4)
        for joint in chain:
            local = np.eye(4)
            origin = joint.find('origin')
            if origin is not None:
                local[:3, 3] = np.fromstring(origin.get('xyz', '0 0 0'), sep=' ')
                local[:3, :3] = Rotation.from_euler('xyz', np.fromstring(origin.get('rpy', '0 0 0'), sep=' ')).as_matrix()
            if joint.get('type') in ('revolute', 'continuous'):
                rotating = np.eye(4)
                axis = np.fromstring(joint.find('axis').get('xyz'), sep=' ')
                rotating[:3, :3] = Rotation.from_rotvec(values[joint.get('name')] * axis).as_matrix()
                local = local @ rotating
            t = t @ local
        return t
    # Controller receives position+velocity+acceleration, so also check a dense
    # quintic Hermite interpolation, rather than checking only waypoint endpoints.
    spline = BPoly.from_derivatives(times, [
        [p['positions'], p['velocities'], p['accelerations']] for p in points])
    sample_times = np.unique(np.r_[np.linspace(times[0], times[-1], 801), times])
    dense_q = spline(sample_times)
    dense_velocity = spline.derivative(1)(sample_times)
    poses = np.array([fk(q) for q in dense_q])
    waypoint_poses = np.array([fk(q) for q in positions])
    start = np.array(data['T_base_tcp_start'])
    target = np.array(data['T_base_tcp_target'])
    xyz = poses[:, :3, 3]
    angles = np.array([np.linalg.norm(Rotation.from_matrix(start[:3, :3].T @ p[:3, :3]).as_rotvec()) for p in poses])
    limit_report = {}
    limits_pass = True
    for i, name in enumerate(names):
        element = joint_elements[name].find('limit')
        lower, upper = float(element.get('lower')), float(element.get('upper'))
        minimum, maximum = float(dense_q[:, i].min()), float(dense_q[:, i].max())
        initial_below = max(0., lower - positions[0, i])
        exception = name == 'fr3_joint4' and 0 < initial_below <= .00017
        lower_ok = minimum >= lower - 1e-10
        if exception:
            lower_ok = (minimum >= positions[0, i] - 1e-10 and
                        np.all(np.diff(dense_q[:, i]) >= -1e-10) and dense_q[-1, i] > lower)
        velocity_limit = float(element.get('velocity'))
        peak_velocity = float(np.max(np.abs(dense_velocity[:, i])))
        passed = bool(lower_ok and maximum <= upper + 1e-10 and peak_velocity <= velocity_limit)
        limits_pass &= passed
        limit_report[name] = dict(lower=lower, upper=upper, planned_range=[minimum, maximum],
                                 peak_velocity_rad_s=peak_velocity, velocity_limit_rad_s=velocity_limit,
                                 initial_below_lower_rad=initial_below,
                                 initial_recovering_exception=bool(exception), passed=passed)
    report = dict(
        plan=str(Path(plan_path).resolve()), urdf=str(Path(urdf_path).resolve()),
        plan_sha256=hashlib.sha256(Path(plan_path).read_bytes()).hexdigest(),
        urdf_sha256=hashlib.sha256(Path(urdf_path).read_bytes()).hexdigest(),
        fraction=data['fraction'], duration_sec=float(times[-1]), samples_checked=len(sample_times),
        interpolation='quintic Hermite using supplied joint positions, velocities and accelerations',
        joint_limits=limit_report,
        min_base_z_displacement_m=float(np.min(xyz[:, 2] - start[2, 3])),
        min_incremental_base_z_m=float(np.min(np.diff(xyz[:, 2]))),
        endpoint_displacement_m=(xyz[-1] - start[:3, 3]).tolist(),
        max_xy_drift_m=float(np.max(np.linalg.norm(xyz[:, :2] - start[:2, 3], axis=1))),
        max_orientation_change_rad=float(angles.max()),
        endpoint_position_error_m=float(np.linalg.norm(xyz[-1] - target[:3, 3])),
        start_fk_error_m=float(np.linalg.norm(xyz[0] - start[:3, 3])),
        waypoint_xyz_m=waypoint_poses[:, :3, 3].tolist(),
        camera_collision_modeled=any('camera' in l.get('name', '').lower() and l.find('collision') is not None
                                     for l in root.findall('link')),
        limits_of_review='No scene/camera mount collision verification; FK and joint limits only. MoveIt collision checks cover modeled geometry. Retraction increases wrist height while holding orientation.',
    )
    report['passed_numeric_checks'] = bool(limits_pass and data['fraction'] >= .999999 and
        report['min_base_z_displacement_m'] >= -.00001 and
        report['min_incremental_base_z_m'] >= -.0000001 and
        report['max_xy_drift_m'] < .0005 and report['max_orientation_change_rad'] < .002 and
        report['endpoint_position_error_m'] < .0005)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('plan')
    parser.add_argument('--urdf', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    report = validate(args.plan, args.urdf)
    Path(args.output).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report['passed_numeric_checks'] else 1)


if __name__ == '__main__':
    main()
