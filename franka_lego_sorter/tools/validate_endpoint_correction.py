"""Independent FK/limits review of a small upward free-space correction."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from validate_lift_plan import validate


def review(path,urdf):
    data=json.loads(Path(path).read_text())
    correction=data['endpoint_correction']
    original=np.asarray(correction['physical_goal'])
    command=np.asarray(data['T_base_tcp_target'])
    start=np.asarray(data['T_base_tcp_start'])
    offset=np.linalg.norm(command[:3,3]-original[:3,3])
    from scipy.spatial.transform import Rotation
    angle=Rotation.from_matrix(original[:3,:3].T@command[:3,:3]).magnitude()
    report=validate(path,urdf)
    report['review_kind']='bounded_upward_endpoint_correction'
    report['physical_goal']=original.tolist()
    report['command_to_goal_offset_m']=float(offset)
    report['command_to_goal_rotation_rad']=float(angle)
    report['passed_numeric_checks']=bool(
        all(j['passed'] for j in report['joint_limits'].values()) and
        data['fraction']>=.999999 and report['duration_sec']>=8 and
        offset<=.003 and angle<=.01 and np.linalg.norm(command[:3,3]-start[:3,3])<=.006 and
        report['min_base_z_displacement_m']>=-.00001 and report['min_incremental_base_z_m']>=-.0000001 and
        report['max_xy_drift_m']<=.006 and report['max_orientation_change_rad']<=.02 and
        report['start_fk_error_m']<=.0005 and report['endpoint_position_error_m']<=.0005)
    report['plan_sha256']=hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('plan');p.add_argument('--urdf',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();r=review(a.plan,a.urdf)
    Path(a.output).write_text(json.dumps(r,indent=2))
    print(json.dumps({k:r[k] for k in ['passed_numeric_checks','duration_sec','endpoint_displacement_m',
        'max_xy_drift_m','max_orientation_change_rad','command_to_goal_offset_m','plan_sha256']},indent=2))
    raise SystemExit(0 if r['passed_numeric_checks'] else 1)
