"""Verify a bounded final placement pose while keeping the Hand closed."""
import hashlib,json
from pathlib import Path
import numpy as np
from .endpoint_servo import pose_error

def verify_review(plan_path,reviewed):
    root=Path(__file__).resolve().parents[1];path=Path(plan_path)
    p=json.loads(path.read_text());m=p['isolated_diagnostic_stage'];checks=reviewed.get('checks',{})
    required={'application_wrist_branch','joint_position_and_speed_limits','complete_planning_fraction','minimum12s_duration','stock_hand_cloud_clear','full_arm_cloud_clear','full_arm_table_clear','camera_bracket_assumed_envelope_clear','recorded_environment_boxes_clear','held_box1mm_above_slot_observed_max','held_box_observed_cloud_clear','held_box_static_input_boxes_clear','approach_cartesian_line','approach_bounded_progress'}
    if m.get('stage')!='first_loose_release_pose'or m.get('gripper_action_authorized')is not False or reviewed.get('purpose')!='isolated_first_loose_release_pose'or reviewed.get('passed_numeric_checks')is not True or not required.issubset(checks)or not all(v is True for v in checks.values())or reviewed.get('failed_checks')or reviewed.get('camera_mode')!='oriented_contact':raise ValueError('Complete final-pose review required; no opening authorized')
    files=[(path,'plan_sha256'),(root/'tools/validate_first_loose_release_pose.py','wrapper_sha256'),(root/'tools/validate_isolated_hover.py','reviewer_sha256'),(root/'lego_sorter/contact_camera_envelope.py','camera_helper_sha256'),(root/'lego_sorter/environment_boxes.py','environment_boxes_module_sha256'),(root/'data/commissioning/fr3_stock_hand.urdf','urdf_sha256'),(Path(reviewed['geometry_report']),'geometry_sha256')]
    for key in ('targets','snapshot','handeye','reference_plan','reference_result'):
        if Path(m[key]).resolve()!=Path(reviewed[key]).resolve():raise ValueError('Release-pose evidence association changed')
        files.append((Path(m[key]),key+'_sha256'))
    files.append((Path(reviewed['environment_boxes']),'environment_boxes_sha256'))
    for filename,key in files:
        if hashlib.sha256(filename.read_bytes()).hexdigest()!=reviewed.get(key):raise ValueError('Release-pose evidence/code changed: '+key)
    target=json.loads(Path(m['targets']).read_text())
    for item in target['source_files']:
        if hashlib.sha256(Path(item['path']).read_bytes()).hexdigest()!=item['sha256']:raise ValueError('Placement source changed')
    if not np.allclose(m['physical_goal'],target['T_base_tcp_release'],atol=1e-10):raise ValueError('Physical target changed')
    if abs(float(m['gripper_width_m'])-float(reviewed['hand_opening_assumed_m']))>1e-9:raise ValueError('Reviewed holding width changed')

def execute_release_pose(robot,plan,meta):
    before=robot.gripper_state()
    if abs(float(before['width_m'])-float(meta['gripper_width_m']))>.0002:raise RuntimeError('Holding width changed before final pose')
    goal=np.array(meta['physical_goal']);actual=np.array(robot.execute_linear_plan(plan,verification_target=goal));after=robot.gripper_state()
    delta,angle=pose_error(goal,actual);t=json.loads(Path(meta['targets']).read_text())
    n=np.array(t['support_plane_base']['normal']);offset=float(t['support_plane_base']['offset_p99_m'])+float(t['minimum_clearance_above_p99_m'])-float(t['minimum_clearance_above_observed_max_m'])
    bottom=np.array(t['estimated_bottom_footprint_tcp_m'])@actual[:3,:3].T+actual[:3,3]
    clearance=float((bottom@n-offset).min());width_ok=abs(float(after['width_m'])-float(before['width_m']))<=.0002
    result=dict(physical_goal=goal.tolist(),measured_after=actual.tolist(),physical_position_error_m=float(np.linalg.norm(delta)),physical_rotation_error_rad=float(angle),minimum_bottom_clearance_above_slot_observed_max_m=clearance,gripper_before=before,gripper_after=after,holding_width_verified=width_ok,release_action_sent=False,production_count_changed=False)
    result['success']=bool(np.linalg.norm(delta)<=.001 and angle<=.01 and clearance>=.001 and width_ok)
    if not result['success']:raise RuntimeError('Final physical placement pose failed; keep fingers closed: '+str(result))
    return result
