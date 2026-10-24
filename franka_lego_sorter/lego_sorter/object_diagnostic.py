"""Hash-pinned measured-object diagnostics; currently only approach30 enabled."""
import hashlib,json
from pathlib import Path
import numpy as np
from .endpoint_servo import pose_error


def bounded_approach_rotation(turn):
 from scipy.spatial.transform import Rotation
 rotation=Rotation.from_matrix(np.asarray(turn,float));vector=rotation.as_rotvec();observed=float(np.linalg.norm(vector))
 if not np.isfinite(observed)or observed>.020+1e-12:raise ValueError('Observed approach orientation exceeds20mrad acquisition bound')
 applied=min(observed,.0095)
 correction=Rotation.from_rotvec(vector*(applied/observed)).as_matrix()if observed>0 else np.eye(3)
 return correction,dict(observed_approach_rotation_rad=observed,applied_rotation_feedforward_rad=applied,rotation_feedforward_saturated=bool(observed>.0095))


def sampling_pose_matches(actual,expected):
 from scipy.spatial.transform import Rotation
 a=np.asarray(actual,float);e=np.asarray(expected,float)
 return bool(np.linalg.norm(a[:3,3]-e[:3,3])<=.00005 and Rotation.from_matrix(e[:3,:3].T@a[:3,:3]).magnitude()<=.0001)


def verify_review(plan_path,reviewed):
 root=Path(__file__).resolve().parents[1];path=Path(plan_path);p=json.loads(path.read_text());m=p['isolated_diagnostic_stage'];checks=reviewed.get('checks',{})
 required={'application_wrist_branch','minimum15s_duration','stock_hand_cloud_clear','full_arm_cloud_clear','full_arm_table_clear','camera_bracket_assumed_envelope_clear','recorded_environment_boxes_clear','object_approach20mm_above_upper','object_approach_straight_line','object_approach_bounded_progress'}
 if m.get('stage')=='object_grasp_pose':
  required.difference_update({'minimum15s_duration','object_approach20mm_above_upper'});required.update({'minimum12s_duration','object_grasp_distal_above_lower'})
 if m.get('stage')not in('object_approach30','object_grasp_pose')or m.get('gripper_action_authorized')is not False or reviewed.get('purpose')!='diagnostic_'+m['stage']or reviewed.get('passed_numeric_checks')is not True or not required.issubset(checks)or not all(v is True for v in checks.values())or reviewed.get('failed_checks'):raise ValueError('Only complete passing measured-object approach review enabled')
 files=[(path,'plan_sha256'),(root/'lego_sorter/object_diagnostic.py','object_diagnostic_module_sha256'),(root/'tools/validate_object_diagnostic.py','wrapper_sha256'),(root/'tools/validate_isolated_hover.py','reviewer_sha256'),(root/'lego_sorter/contact_camera_envelope.py','camera_helper_sha256'),(root/'lego_sorter/environment_boxes.py','environment_boxes_module_sha256'),(root/'data/commissioning/fr3_stock_hand.urdf','urdf_sha256'),(Path(reviewed['geometry_report']),'geometry_sha256')]
 for key in('object_target','scene_snapshot','handeye','environment_boxes')+(('reference_plan','reference_result')if m['stage']=='object_grasp_pose'else()):
  if Path(m[key]).resolve()!=Path(reviewed[key]).resolve():raise ValueError('Object diagnostic evidence path changed')
  files.append((Path(m[key]),key+'_sha256'))
 for filename,key in files:
  if hashlib.sha256(filename.read_bytes()).hexdigest()!=reviewed.get(key):raise ValueError('Object diagnostic evidence/code changed: '+key)
 if m['stage']=='object_grasp_pose'and(not np.allclose(m['physical_goal'],reviewed['physical_goal'],atol=1e-10)or m['reference_execution_success']is not reviewed['reference_execution_success']):raise ValueError('Physical goal/reference outcome changed')
 for source in json.loads(Path(m['object_target']).read_text())['source_files']:
  if hashlib.sha256(Path(source['path']).read_bytes()).hexdigest()!=source['sha256']:raise ValueError('Measured object source changed')


def execute_stage(robot,plan,meta):
 if meta.get('stage')not in('object_approach30','object_grasp_pose'):raise ValueError('Only object approach/grasp-pose enabled')
 before=robot.gripper_state()
 if abs(float(before['width_m'])-float(meta['gripper_width_m']))>.0002:raise RuntimeError('Object approach opening changed')
 actual=robot.execute_linear_plan(plan,verification_target=np.array(meta['physical_goal']))if meta['stage']=='object_grasp_pose'else robot.execute_linear_plan(plan);after=robot.gripper_state()
 if abs(float(after['width_m'])-float(before['width_m']))>.0002:raise RuntimeError('Hand opening changed during object approach')
 result=dict(measured_after=actual.tolist(),gripper_before=before,gripper_after=after,gripper_action_sent=False,lift_action_sent=False,production_count_changed=False)
 if meta['stage']=='object_grasp_pose':
  goal=np.array(meta['physical_goal']);delta,angle=pose_error(goal,actual);t=json.loads(Path(meta['object_target']).read_text());n=np.array(t['normal_base']);lower=float(t['visible_upper_p99_offset_base_m'])-float(t['toe_center_insertion_below_observed_p99_m'])-float(t['minimum_toe_clearance_above_prior_lower_layer_globalmax_m']);toe=np.array([[x,y,.0095]for x in(-.011,.011)for y in(-.05,.05)]);clearance=float(((toe@actual[:3,:3].T+actual[:3,3])@n-lower).min())
  result.update(physical_goal=goal.tolist(),physical_position_error_m=float(np.linalg.norm(delta)),physical_rotation_error_rad=float(angle),minimum_toe_above_prior_lower_layer_m=clearance,reference_execution_success=meta['reference_execution_success'])
  if np.linalg.norm(delta)>.001 or angle>.01 or clearance<.003:raise RuntimeError('Measured object grasp pose failed strict verification; no finger action: '+str(result))
 return result
