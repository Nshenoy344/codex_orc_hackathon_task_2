#!/usr/bin/env python3
"""Offline analysis of captured JTC/Franka state. No hardware imports or writes."""
import argparse,json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation

def main():
    p=argparse.ArgumentParser();p.add_argument('samples');p.add_argument('--urdf',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    data=json.loads(Path(a.samples).read_text());c=data['controller'];robot=data['robot']
    if not c or not robot:raise ValueError('Need controller and robot samples')
    r=ET.parse(a.urdf).getroot();bychild={j.find('child').get('link'):j for j in r.findall('joint')}
    chain=[];child='fr3_hand_tcp'
    while child!='fr3_link0':
        j=bychild[child];chain.append(j);child=j.find('parent').get('link')
    chain.reverse()
    def fk(names,q):
        vals=dict(zip(names,q));t=np.eye(4)
        for j in chain:
            x=np.eye(4);o=j.find('origin')
            if o is not None:
                x[:3,3]=np.fromstring(o.get('xyz','0 0 0'),sep=' ')
                x[:3,:3]=Rotation.from_euler('xyz',np.fromstring(o.get('rpy','0 0 0'),sep=' ')).as_matrix()
            if j.get('type') in ('revolute','continuous'):
                qx=np.eye(4);qx[:3,:3]=Rotation.from_rotvec(vals[j.get('name')]*np.fromstring(j.find('axis').get('xyz'),sep=' ')).as_matrix();x=x@qx
            t=t@x
        return t
    names=c[-1]['joint_names'];refs=np.array([v['reference']['positions'] for v in c]);acts=np.array([v['feedback']['positions'] for v in c]);err=refs-acts
    rf=np.array([fk(names,q) for q in refs]);af=np.array([fk(names,q) for q in acts])
    cart=rf[:,:3,3]-af[:,:3,3]
    angle=np.array([np.linalg.norm(Rotation.from_matrix(x[:3,:3].T@y[:3,:3]).as_rotvec()) for x,y in zip(rf,af)])
    wrench=[]
    for v in robot:
        w=v['o_f_ext_hat_k']['wrench'];wrench.append([w['force'][k] for k in ('x','y','z')]+[w['torque'][k] for k in ('x','y','z')])
    wrench=np.array(wrench)
    out=dict(controller_sample_count=len(c),robot_sample_count=len(robot),joint_names=names,
        reference_q_std_rad=refs.std(axis=0).tolist(),measured_q_std_rad=acts.std(axis=0).tolist(),
        mean_joint_error_rad=err.mean(axis=0).tolist(),std_joint_error_rad=err.std(axis=0).tolist(),
        mean_controller_effort_nm=np.mean([v['output']['effort'] for v in c],axis=0).tolist(),
        mean_measured_fk_tcp_m=af[:,:3,3].mean(axis=0).tolist(),
        mean_reference_fk_tcp_m=rf[:,:3,3].mean(axis=0).tolist(),
        mean_cartesian_target_minus_measured_m=cart.mean(axis=0).tolist(),
        cartesian_error_norm_mean_m=float(np.linalg.norm(cart,axis=1).mean()),
        cartesian_error_norm_std_m=float(np.linalg.norm(cart,axis=1).std()),
        orientation_error_mean_rad=float(angle.mean()),orientation_error_std_rad=float(angle.std()),
        mean_external_wrench_base_N_Nm=wrench.mean(axis=0).tolist(),std_external_wrench_base_N_Nm=wrench.std(axis=0).tolist(),
        ee_mass_kg=robot[-1]['inertia_ee']['inertia']['m'],load_mass_kg=robot[-1]['inertia_load']['inertia']['m'],
        current_robot_mode=robot[-1]['robot_mode'],any_current_errors=any(any(v['current_errors'].values()) for v in robot),
        conclusion='Stable real joint-space tracking bias under torque PD; not stale TCP TF. Cause of model/load/cable residual is not uniquely identified by one pose.',
        recommendations=[
            'Do not inflate placement tolerance to declare a tight physical transfer verified.',
            'Independently measure added camera/bracket mass and center of mass before altering gravity/load model; do not infer mass solely from one external-wrench reading.',
            'A measured-pose closed-loop correction can be tested at free-space commissioning height: recompute a small endpoint correction, replan from measured state with the existing collision checks, bound per-correction translation/rotation and total corrections, and require strict measured-pose convergence.',
            'Calibration observations should retain actual stable measured TCP rather than assuming the commanded endpoint was reached exactly.',
            'No controller parameter changes, compensation commands, recovery, or motion were made during this read-only review.'])
    Path(a.output).write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
