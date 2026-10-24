"""Review one bias-compensated isolated loose-brick gripper pose; no finger action."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from lego_sorter.geometry import matrix
from validate_isolated_hover import review as review_path


def contract(plan_path):
    path=Path(plan_path);p=json.loads(path.read_text());m=p['isolated_diagnostic_stage']
    if m.get('stage')!='grasp_pose':raise ValueError('Only isolated diagnostic grasp_pose is supported')
    if abs(float(m['gripper_width_m'])-.030)>.0002:raise ValueError('Grasp-pose move requires30mm measured opening')
    src=json.loads(Path(m['targets']).read_text());plane=next(v for v in src['planes']if v['plane']==m['plane'])
    if src.get('kind')!='explicit_nominal_geometry_diagnostic_only'or src.get('production_qualified')is not False:raise ValueError('Only explicit loose-brick nominal diagnostic geometry supported')
    for key,source_key,hash_key in [('snapshot','snapshot','snapshot_sha256'),('handeye','handeye_artifact','handeye_sha256')]:
        if Path(m[key]).resolve()!=Path(src[source_key]).resolve()or hashlib.sha256(Path(m[key]).read_bytes()).hexdigest()!=src[hash_key]:raise ValueError('Diagnostic source evidence changed')
    physical=matrix(m['physical_goal']);start=matrix(p['T_base_tcp_start']);command=matrix(p['T_base_tcp_target'])
    if not np.allclose(physical,plane['poses']['grasp'],atol=1e-8):raise ValueError('Physical goal differs from pinned diagnostic grasp')
    ap=json.loads(Path(m['bias_reference_plan']).read_text());ar=json.loads(Path(m['bias_reference_result']).read_text())
    if ar.get('success')is not True or Path(ar['plan_file']).resolve()!=Path(m['bias_reference_plan']).resolve():raise ValueError('Approach bias lacks successful physical reference')
    approach=matrix(ap['T_base_tcp_target']);observed=matrix(ar['T_base_tcp_after'])
    if not np.allclose(approach,plane['poses']['hover30'],atol=1e-8):raise ValueError('Bias reference is not this brick approach30')
    if np.linalg.norm(start[:3,3]-observed[:3,3])>.001 or Rotation.from_matrix(start[:3,:3].T@observed[:3,:3]).magnitude()>.01:raise ValueError('Robot moved since successful approach checkpoint')
    delta=approach[:3,3]-start[:3,3];turn=approach[:3,:3]@start[:3,:3].T
    if np.linalg.norm(delta)>.003 or Rotation.from_matrix(turn).magnitude()>.01:raise ValueError('Measured approach bias exceeds3mm/.01rad')
    expected=physical.copy();expected[:3,3]+=delta;expected[:3,:3]=turn@physical[:3,:3]
    if not np.allclose(command,expected,atol=1e-8):raise ValueError('Command must equal exact bounded measured-bias compensation')
    if not 0<start[2,3]-command[2,3]<=.035 or np.linalg.norm(start[:3,3]-command[:3,3])>.035:raise ValueError('Grasp-pose move exceeds35mm descending segment')
    return p,m,src,plane,physical


def review(plan_path):
    path=Path(plan_path);p,m,src,plane,physical=contract(path);measured=plane['measured_table_plane'];geometry=path.with_suffix('.scene_geometry.json')
    g={'visible_top_base_m':plane['assumed_nominal_stud_top_base_m'],'table_point_base_m':plane['floor_ray_point_base_m'],'measured_table_plane':{'normal_base':measured['normal_base'],**measured['quality']},'source_targets':str(Path(m['targets']).resolve()),'source_targets_sha256':hashlib.sha256(Path(m['targets']).read_bytes()).hexdigest(),'scene_reference_semantics':'Measured localfloor plus explicit nominal LEGO dimensions; not measured brickbody height.'}
    if geometry.exists()and json.loads(geometry.read_text())!=g:raise ValueError('Existing grasp-pose geometry differs from pinned diagnostic source')
    geometry.write_text(json.dumps(g,indent=2))
    r=review_path(path,'data/commissioning/fr3_stock_hand.urdf',m['snapshot'],m['handeye'],geometry,joint_space=True,opening=float(m['gripper_width_m']),camera_mode='oriented_contact')
    checks=r['checks'];checks.pop('minimum150mm_tip_clearance');checks.update(command_tip_table_clearance=r['minimum_modeled_tip_table_clearance_m']>=.003,
        grasp_pose_straight_line=r['max_cartesian_line_deviation_m']<=.0005,
        grasp_pose_progress=r['path_progress_min_m']>=-.0005 and r['path_progress_max_m']<=r['path_length_m']+.0005)
    normal=np.array(measured['normal_base']);floor=np.array(plane['floor_ray_point_base_m']);corners=np.array([[x,y,.0095]for x in(-.011,.011)for y in(-.05,.05)])
    # Command orientation stays within.01rad of the vertical physical goal.
    # Conservative full distal footprint also remains above the support plane.
    end=matrix(p['T_base_tcp_target']);clearance=float(((corners@end[:3,:3].T+end[:3,3]-floor)@normal).min())
    checks['distal_command_footprint_table_clearance']=clearance>=.003;r['minimum_command_distal_footprint_table_clearance_m']=clearance
    r['checks']={k:bool(v)for k,v in checks.items()};r['failed_checks']=[k for k,v in r['checks'].items()if not v];r['passed_numeric_checks']=not r['failed_checks'];r['purpose']='isolated_grasp_pose'
    r['physical_goal']=physical.tolist();r['strict_physical_position_tolerance_m']=.001;r['strict_physical_rotation_tolerance_rad']=.01
    for key in('targets','bias_reference_plan','bias_reference_result'):
        r[key]=str(Path(m[key]).resolve());r[key+'_sha256']=hashlib.sha256(Path(m[key]).read_bytes()).hexdigest()
    r['wrapper_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();r['limits_of_review']='Isolated loose-brick pose only; nominal LEGO geometry and measuredfloor. No closure, lift or sorting action authorized by this stage.'
    return r


def main():
    p=argparse.ArgumentParser();p.add_argument('plan');a=p.parse_args();r=review(a.plan);Path(a.plan).with_suffix('.validation.json').write_text(json.dumps(r,indent=2));print(json.dumps({k:v for k,v in r.items()if k not in('joint_limits','full_stock_hand_observed_cloud_sweep','full_arm_convex_hull_review')},indent=2));return 0 if r['passed_numeric_checks'] else 2
if __name__=='__main__':raise SystemExit(main())
