"""Hash-pinned vertical empty-Hand retreat after one diagnostic release."""
import hashlib,json
from pathlib import Path
import numpy as np


def verify_review(plan_path,reviewed):
 root=Path(__file__).resolve().parents[1];path=Path(plan_path);p=json.loads(path.read_text());m=p['isolated_diagnostic_stage'];checks=reviewed.get('checks',{})
 required={'application_wrist_branch','minimum15s_duration','stock_hand_cloud_clear','full_arm_cloud_clear','full_arm_table_clear','camera_bracket_assumed_envelope_clear','recorded_environment_boxes_clear','released_static_brick_hand_clear','retreat_vertical_line','retreat_bounded_progress','retreat_monotonic_up'}
 if m.get('stage')!='postrelease_retreat350'or m.get('gripper_action_authorized')is not False or m.get('attached_model_removed')is not True or reviewed.get('purpose')!='isolated_postrelease_retreat350'or reviewed.get('passed_numeric_checks')is not True or not required.issubset(checks)or not all(v is True for v in checks.values())or reviewed.get('failed_checks')or reviewed.get('camera_mode')!='oriented_contact':raise ValueError('Complete passing static-object postrelease retreat required')
 files=[(path,'plan_sha256'),(root/'tools/validate_postrelease_retreat.py','wrapper_sha256'),(root/'tools/validate_isolated_hover.py','reviewer_sha256'),(root/'lego_sorter/contact_camera_envelope.py','camera_helper_sha256'),(root/'lego_sorter/released_brick_geometry.py','released_geometry_module_sha256'),(root/'lego_sorter/environment_boxes.py','environment_boxes_module_sha256'),(root/'data/commissioning/fr3_stock_hand.urdf','urdf_sha256'),(Path(reviewed['geometry_report']),'geometry_sha256')]
 for key in('release_result','placement_target','scene_snapshot','handeye','environment_boxes'):
  if Path(m[key]).resolve()!=Path(reviewed[key]).resolve():raise ValueError('Retreat evidence association changed')
  files.append((Path(m[key]),key+'_sha256'))
 for filename,key in files:
  if hashlib.sha256(filename.read_bytes()).hexdigest()!=reviewed.get(key):raise ValueError('Retreat evidence/code changed: '+key)


def execute_retreat(robot,plan,meta):
 before_hand=robot.gripper_state()
 if abs(float(before_hand['width_m'])-.020)>.0002 or abs(float(before_hand['width_m'])-float(meta['gripper_width_m']))>.0002:raise RuntimeError('Fresh20mm opening changed; no retreat')
 before=np.array(robot.current_pose());actual=np.array(robot.execute_linear_plan(plan));after=robot.gripper_state()
 if abs(float(after['width_m'])-.020)>.0002:raise RuntimeError('Hand opening changed during retreat')
 rise=float(actual[2,3]-before[2,3]);expected=float(plan.target_pose[2,3]-plan.start_pose[2,3])
 if rise<expected-.003:raise RuntimeError('Measured upward retreat insufficient; inspect actual pose')
 return dict(measured_after=actual.tolist(),measured_upward_retreat_m=rise,gripper_before=before_hand,gripper_after=after,opening_width_verified=True,attached_model_removed=True,static_released_brick_model=True,gripper_action_sent=False,release_action_sent=False,brick_placement_visually_verified=False,production_count_changed=False)
