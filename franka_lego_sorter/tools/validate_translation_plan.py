#!/usr/bin/env python3
"""Offline exact-URDF review of the bounded raised base-X board/source viewing shifts.

Reuses the dense quintic Hermite FK/joint-limit method from validate_lift_plan.
It sends no robot commands. Unmodeled camera/bracket/environment collision
geometry is not proved clear by this numerical review.
"""
import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from scipy.interpolate import BPoly
from scipy.spatial.transform import Rotation


def validate(plan_path, urdf_path):
    data=json.loads(Path(plan_path).read_text())
    root=ET.parse(urdf_path).getroot()
    joints={j.get('name'):j for j in root.findall('joint')}
    bychild={j.find('child').get('link'):j for j in root.findall('joint')}
    chain=[];child=data['tcp_frame']
    while child!=data['base_frame']:
        if child not in bychild:raise ValueError('Base/TCP chain is absent from supplied URDF')
        joint=bychild[child];chain.append(joint);child=joint.find('parent').get('link')
    chain.reverse()
    names=data['joint_names'];points=data['points']
    if len(points)<2 or not names or len(set(names))!=len(names):
        raise ValueError('Need a nonempty uniquely named timed trajectory')
    times=np.array([p['time_sec'] for p in points],float)
    if not np.isfinite(times).all() or times[0]<0 or np.any(np.diff(times)<=0):
        raise ValueError('Invalid trajectory times')
    for point in points:
        for field in ('positions','velocities','accelerations'):
            values=np.asarray(point[field],float)
            if values.shape!=(len(names),) or not np.isfinite(values).all():
                raise ValueError('Malformed/nonfinite trajectory joint arrays')
    spline=BPoly.from_derivatives(times,[[p['positions'],p['velocities'],p['accelerations']] for p in points])
    sample_times=np.unique(np.r_[times,np.linspace(times[0],times[-1],2001)])
    dense=spline(sample_times);vel=spline.derivative(1)(sample_times)
    def fk(q):
        values=dict(zip(names,q));result=np.eye(4)
        for joint in chain:
            local=np.eye(4);origin=joint.find('origin')
            if origin is not None:
                local[:3,3]=np.fromstring(origin.get('xyz','0 0 0'),sep=' ')
                local[:3,:3]=Rotation.from_euler('xyz',np.fromstring(origin.get('rpy','0 0 0'),sep=' ')).as_matrix()
            kind=joint.get('type')
            if kind in ('revolute','continuous','prismatic'):
                name=joint.get('name')
                if name not in values:raise ValueError(f'Trajectory omits chain joint {name}')
                axis=np.fromstring(joint.find('axis').get('xyz'),sep=' ')
                moving=np.eye(4)
                if kind=='prismatic':moving[:3,3]=values[name]*axis
                else:moving[:3,:3]=Rotation.from_rotvec(values[name]*axis).as_matrix()
                local=local@moving
            result=result@local
        return result
    poses=np.array([fk(q) for q in dense]);xyz=poses[:,:3,3]
    start=np.asarray(data['T_base_tcp_start'],float);target=np.asarray(data['T_base_tcp_target'],float)
    if (start.shape!=(4,4) or target.shape!=(4,4) or not np.isfinite(start).all() or
            not np.isfinite(target).all()):raise ValueError('Malformed TCP endpoints')
    angles=np.array([np.linalg.norm(Rotation.from_matrix(start[:3,:3].T@p[:3,:3]).as_rotvec()) for p in poses])
    limits={};limits_pass=True
    for index,name in enumerate(names):
        if name not in joints:raise ValueError(f'Unknown trajectory joint {name}')
        element=joints[name].find('limit')
        lower,upper,velocity_limit=map(float,(element.get('lower'),element.get('upper'),element.get('velocity')))
        minimum,maximum=float(dense[:,index].min()),float(dense[:,index].max())
        peak=float(np.max(np.abs(vel[:,index])))
        passed=minimum>=lower-1e-9 and maximum<=upper+1e-9 and peak<=velocity_limit
        limits_pass &= passed
        limits[name]=dict(lower=lower,upper=upper,planned_range=[minimum,maximum],
                          peak_velocity_rad_s=peak,velocity_limit_rad_s=velocity_limit,passed=bool(passed))
    cross=np.linalg.norm(xyz[:,1:]-start[1:3,3],axis=1)
    displacement=target[:3,3]-start[:3,3]
    requested_distance=float(data.get('commissioning_translation',{}).get('distance_m',.20))
    if not np.isfinite(requested_distance) or not 0<abs(requested_distance)<=.20:
        raise ValueError('Viewing shift magnitude must be positive and at most20cm')
    direction=1. if requested_distance>0 else -1.
    negative_policy=None
    if requested_distance<0:
        intent=data.get('commissioning_translation',{})
        if intent.get('purpose')!='return_to_observed_source' or intent.get('axis')!='x':
            raise ValueError('NegativeX movement requires explicit observed-source return policy')
        snapshot=Path(intent['source_reference_snapshot'])
        if hashlib.sha256(snapshot.read_bytes()).hexdigest()!=intent['source_reference_sha256']:
            raise ValueError('Observed source snapshot changed after return planning')
        with np.load(snapshot,allow_pickle=False)as observation:
            source=np.asarray(observation['T_base_tcp'],float)
            for key,bound in [('robot_drift_m',.001),('robot_drift_rad',.005)]:
                if key not in observation or not np.isfinite(float(observation[key])) or float(observation[key])>bound:
                    raise ValueError('Observed source intent is not stationary')
        if source.shape!=(4,4) or not np.isfinite(source).all() or not np.allclose(source,intent['T_base_tcp_source_reference'],atol=1e-8):
            raise ValueError('Source reference pose differs from pinned measured observation')
        for name,pose in [('start',start),('target',target),('source',source)]:
            if not (.20<=pose[0,3]<=.60 and -.15<=pose[1,3]<=.15 and .33<=pose[2,3]<=.40):
                raise ValueError(f'{name} pose leaves the observed raised traversal corridor')
        if (abs(source[2,3]-start[2,3])>.02 or abs(source[1,3]-start[1,3])>.03 or
            target[0,3]<source[0,3]-.005 or
            abs(target[0,3]-source[0,3])>=abs(start[0,3]-source[0,3])):
            raise ValueError('NegativeX return does not approach its observed source corridor')
        negative_policy=dict(purpose=intent['purpose'],source_reference_snapshot=str(snapshot),
                             T_base_tcp_source_reference=source.tolist())
    directed=direction*(xyz[:,0]-start[0,3])
    minimum_duration=max(8.,100.*abs(requested_distance))
    report=dict(plan=str(Path(plan_path).resolve()),urdf=str(Path(urdf_path).resolve()),
        plan_sha256=hashlib.sha256(Path(plan_path).read_bytes()).hexdigest(),
        urdf_sha256=hashlib.sha256(Path(urdf_path).read_bytes()).hexdigest(),
        fraction=float(data['fraction']),duration_sec=float(times[-1]),samples_checked=len(sample_times),
        interpolation='quintic Hermite from commanded position,velocity,acceleration',
        target_displacement_m=displacement.tolist(),
        requested_orientation_change_rad=float(np.linalg.norm(Rotation.from_matrix(start[:3,:3].T@target[:3,:3]).as_rotvec())),
        endpoint_displacement_m=(xyz[-1]-start[:3,3]).tolist(),
        min_base_x_displacement_m=float(np.min(xyz[:,0]-start[0,3])),
        min_incremental_base_x_m=float(np.min(np.diff(xyz[:,0]))),
        min_directed_progress_m=float(np.min(directed)),
        min_incremental_directed_progress_m=float(np.min(np.diff(directed))),
        observed_source_return_policy=negative_policy,minimum_tcp_z_m=float(poses[:,2,3].min()),
        max_cross_axis_drift_m=float(cross.max()),max_orientation_change_rad=float(angles.max()),
        endpoint_position_error_m=float(np.linalg.norm(xyz[-1]-target[:3,3])),
        start_fk_error_m=float(np.linalg.norm(xyz[0]-start[:3,3])),joint_limits=limits,
        waypoint_xyz_m=[fk(np.asarray(p['positions']))[:3,3].tolist() for p in points],
        camera_collision_modeled=any('camera' in link.get('name','').lower() and link.find('collision') is not None for link in root.findall('link')),
        limits_of_review='Only exact URDF joint limits, velocities, and TCP path geometry. Camera/bracket and scene collision geometry remain unmodeled; inspect the actual swept workspace.')
    report['passed_numeric_checks']=bool(limits_pass and report['fraction']>=.999999 and
        np.allclose(displacement,[requested_distance,0,0],atol=1e-5,rtol=0) and
        report['requested_orientation_change_rad']<=1e-5 and
        report['duration_sec']>=minimum_duration and report['min_directed_progress_m']>=-.00001 and
        report['min_incremental_directed_progress_m']>=-1e-7 and report['max_cross_axis_drift_m']<=.0005 and
        report['max_orientation_change_rad']<=.002 and report['endpoint_position_error_m']<=.0005 and
        report['start_fk_error_m']<=.0005 and
        (requested_distance>0 or report['minimum_tcp_z_m']>=.3295))
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('plan');p.add_argument('--urdf',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();report=validate(a.plan,a.urdf)
    Path(a.output).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    return 0 if report['passed_numeric_checks'] else 2


if __name__=='__main__':raise SystemExit(main())
