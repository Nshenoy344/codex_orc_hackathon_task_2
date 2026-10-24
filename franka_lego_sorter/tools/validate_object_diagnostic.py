"""Independent reusable measured-object approach review; no closure or lift."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from lego_sorter.environment_boxes import load_boxes
from validate_isolated_hover import review as review_path


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def review(path):
 path=Path(path);plan=json.loads(path.read_text());m=plan['isolated_diagnostic_stage'];target=json.loads(Path(m['object_target']).read_text())
 if m.get('stage')not in('object_approach30','object_grasp_pose')or m.get('gripper_action_authorized')is not False or target.get('production_ready')is not False or target.get('kind')!='isolated_misplaced_brick_recovery_target_only':raise ValueError('Only measured isolated-object approach30 enabled')
 for source in target['source_files']:
  if digest(source['path'])!=source['sha256']:raise ValueError('Measured object source changed')
 if Path(m['scene_snapshot']).resolve()!=Path(target['snapshot']).resolve()or abs(float(m['gripper_width_m'])-float(target['gripper_opening_m']))>.0002:raise ValueError('Object scene/opening differs from measured target')
 expected_handeye=next(Path(v['path']).resolve()for v in target['source_files']if Path(v['path']).suffix=='.yaml')
 if Path(m['handeye']).resolve()!=expected_handeye:raise ValueError('Object handeye differs from measured target evidence')
 start=np.array(plan['T_base_tcp_start']);goal=np.array(plan['T_base_tcp_target'])
 if m['stage']=='object_grasp_pose':
  physical=np.array(m['physical_goal']);ref=json.loads(Path(m['reference_plan']).read_text());res=json.loads(Path(m['reference_result']).read_text())
  allowed=res.get('success')is True or(res.get('success')is False and res.get('status')=='execution_or_verification_failed'and res.get('reason','').startswith('Pose verification failed:'))
  if not allowed or Path(res['plan_file']).resolve()!=Path(m['reference_plan']).resolve():raise ValueError('Unsupported/unknown approach execution outcome')
  rm=ref['isolated_diagnostic_stage']
  if rm.get('stage')!='object_approach30'or Path(rm['object_target']).resolve()!=Path(m['object_target']).resolve()or not np.allclose(ref['T_base_tcp_target'],target['T_base_tcp_approach30'],atol=1e-9):raise ValueError('Bias reference is not this measured object approach')
  if m.get('reference_execution_success')is not bool(res.get('success')):raise ValueError('Prior failed-result status must remain explicit')
  previous=np.array(res['T_base_tcp_after']);approach=np.array(target['T_base_tcp_approach30']);delta=approach[:3,3]-start[:3,3];turn=approach[:3,:3]@start[:3,:3].T
  if np.linalg.norm(start[:3,3]-previous[:3,3])>.003 or Rotation.from_matrix(start[:3,:3].T@previous[:3,:3]).magnitude()>.01 or np.linalg.norm(delta)>.003:raise ValueError('Fresh measured approach translation/recorded-state envelope exceeded')
  from lego_sorter.object_diagnostic import bounded_approach_rotation,sampling_pose_matches
  turn,rotation_diagnostics=bounded_approach_rotation(turn)
  if not np.allclose(physical,target['T_base_tcp_grasp'],atol=1e-9):raise ValueError('Physical grasp pose differs from measured target')
  expected=physical.copy();expected[:3,3]+=delta;expected[:3,:3]=turn@physical[:3,:3]
  if not sampling_pose_matches(goal,expected)or np.linalg.norm(goal[:3,3]-start[:3,3])>.035:raise ValueError('Grasp command differs from bounded local-bias compensation')
 elif not np.allclose(goal,target['T_base_tcp_approach30'],atol=1e-9):raise ValueError('Object approach target changed')
 if not 0<np.linalg.norm(start[:3,3]-goal[:3,3])<=.4 or goal[2,3]>=start[2,3]:raise ValueError('Object approach requires descending segment≤400mm')
 n=np.array(target['normal_base']);point=np.array(target['observed_face_center_base_m']);upper=float(target['visible_upper_p99_offset_base_m']);lower=upper-float(target['toe_center_insertion_below_observed_p99_m'])-float(target['minimum_toe_clearance_above_prior_lower_layer_globalmax_m'])
 geometry=path.with_suffix('.scene_geometry.json');g=dict(visible_top_base_m=(point+n*(upper-point@n)).tolist(),table_point_base_m=(point+n*(lower-point@n)).tolist(),measured_table_plane=dict(normal_base=n.tolist()),surface_reference='Actual visible-upperp99; NOT body/stud classification. Lower reference is prior observed lower-layer globalmaximum.')
 geometry.write_text(json.dumps(g,indent=2));r=review_path(path,'data/commissioning/fr3_stock_hand.urdf',m['scene_snapshot'],m['handeye'],geometry,joint_space=True,opening=float(m['gripper_width_m']),camera_mode='oriented_contact',environment_boxes=load_boxes(m['environment_boxes']),include_dense_poses=m['stage']=='object_grasp_pose')
 checks=r['checks'];checks.pop('minimum150mm_tip_clearance');checks.update(object_approach_straight_line=r['max_cartesian_line_deviation_m']<=.0005,object_approach_bounded_progress=r['path_progress_min_m']>=-.0005 and r['path_progress_max_m']<=r['path_length_m']+.0005)
 if m['stage']=='object_grasp_pose':
  poses=np.array(r.pop('_dense_tcp_poses'));toe=np.array([[x,y,.0095]for x in(-.011,.011)for y in(-.05,.05)]);minimum=min(float(((toe@p[:3,:3].T+p[:3,3])@n-lower).min())for p in poses)
  checks['object_grasp_distal_above_lower']=minimum>=.003;r['minimum_distal_toe_above_prior_lower_layer_m']=minimum;r['physical_goal']=m['physical_goal'];r['reference_execution_success']=m['reference_execution_success'];r.update(rotation_diagnostics)
 else:checks.update(minimum15s_duration=r['duration_sec']>=15.,object_approach20mm_above_upper=r['minimum_modeled_tip_height_above_measured_brick_m']>=.020)
 r.update(purpose='diagnostic_'+m['stage'],checks={k:bool(v)for k,v in checks.items()},wrapper_sha256=digest(__file__),object_diagnostic_module_sha256=digest(Path(__file__).resolve().parents[1]/'lego_sorter/object_diagnostic.py'),environment_boxes_module_sha256=digest(Path(__file__).resolve().parents[1]/'lego_sorter/environment_boxes.py'),limits_of_review='Measured visible-surface approach or locally bias-compensated grasp-pose only; strict physical endpoint/toe clearance required before any closure. No body/stud inference, closure, lift or production count.')
 for key in('object_target','scene_snapshot','handeye','environment_boxes')+(('reference_plan','reference_result')if m['stage']=='object_grasp_pose'else()):r[key]=m[key];r[key+'_sha256']=digest(m[key])
 r['failed_checks']=[k for k,v in r['checks'].items()if not v];r['passed_numeric_checks']=not r['failed_checks'];return r

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('plan');a=p.parse_args();r=review(a.plan);Path(a.plan).with_suffix('.validation.json').write_text(json.dumps(r,indent=2));print(json.dumps({k:v for k,v in r.items()if k not in('joint_limits','full_stock_hand_observed_cloud_sweep','full_arm_convex_hull_review')},indent=2));raise SystemExit(0 if r['passed_numeric_checks']else 2)
