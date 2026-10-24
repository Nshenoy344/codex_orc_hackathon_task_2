"""Strict verification of one reviewed isolated gripper-pose move; no fingers/lift."""
import hashlib,json
from pathlib import Path
import numpy as np
from .geometry import matrix
from .endpoint_servo import pose_error


def verify_review(plan_path,reviewed,repository_root=None):
    root=Path(repository_root)if repository_root else Path(__file__).resolve().parents[1]
    path=Path(plan_path);plan=json.loads(path.read_text());meta=plan.get('isolated_diagnostic_stage',{})
    required={'application_wrist_branch','joint_position_and_speed_limits','complete_planning_fraction','minimum12s_duration','command_tip_table_clearance','distal_command_footprint_table_clearance','grasp_pose_straight_line','grasp_pose_progress','stock_hand_cloud_clear','full_arm_cloud_clear','full_arm_table_clear','camera_bracket_assumed_envelope_clear'}
    checks=reviewed.get('checks',{})
    if meta.get('stage')!='grasp_pose'or reviewed.get('purpose')!='isolated_grasp_pose'or reviewed.get('passed_numeric_checks')is not True or not required.issubset(checks)or not all(v is True for v in checks.values())or reviewed.get('failed_checks')or reviewed.get('camera_mode')!='oriented_contact':raise ValueError('Only complete passing isolated grasp_pose review may execute')
    if abs(float(meta['gripper_width_m'])-.030)>.0002 or abs(float(meta['gripper_width_m'])-float(reviewed['hand_opening_assumed_m']))>1e-9:raise ValueError('Grasp pose requires reviewed30mm opening')
    if not np.allclose(matrix(meta['physical_goal']),matrix(reviewed['physical_goal']),atol=1e-10):raise ValueError('Physical verification goal changed')
    files=[(path,'plan_sha256'),(root/'tools/validate_isolated_hover.py','reviewer_sha256'),(root/'tools/validate_isolated_grasp_pose.py','wrapper_sha256'),(root/'lego_sorter/contact_camera_envelope.py','camera_helper_sha256'),(root/'data/commissioning/fr3_stock_hand.urdf','urdf_sha256'),(Path(reviewed['geometry_report']),'geometry_sha256')]
    for key in('snapshot','handeye','targets','bias_reference_plan','bias_reference_result'):
        if Path(meta[key]).resolve()!=Path(reviewed[key]).resolve():raise ValueError('Grasp-pose evidence path changed: '+key)
        files.append((Path(reviewed[key]),key+'_sha256'))
    for filename,key in files:
        if hashlib.sha256(filename.read_bytes()).hexdigest()!=reviewed.get(key):raise ValueError('Grasp-pose review input/code changed: '+key)
    return True


def execute_grasp_pose(robot,plan,meta):
    if meta.get('stage')!='grasp_pose':raise ValueError('Only the isolated grasp_pose stage is enabled')
    goal=matrix(meta['physical_goal'])
    actual=matrix(robot.execute_linear_plan(plan,verification_target=goal))
    delta,angle=pose_error(goal,actual)
    source=json.loads(Path(meta['targets']).read_text());plane=next(p for p in source['planes']if p['plane']==meta['plane'])
    normal=np.array(plane['measured_table_plane']['normal_base']);floor=np.array(plane['floor_ray_point_base_m'])
    footprint=np.array([[x,y,.0095]for x in(-.011,.011)for y in(-.05,.05)])
    clearance=float(((footprint@actual[:3,:3].T+actual[:3,3]-floor)@normal).min())
    result=dict(physical_goal=goal.tolist(),measured_after=actual.tolist(),physical_position_error_m=float(np.linalg.norm(delta)),physical_rotation_error_rad=float(angle),minimum_measured_tcp_footprint_table_clearance_m=clearance,gripper_action_sent=False,lift_action_sent=False)
    result['success']=bool(np.linalg.norm(delta)<=.001 and angle<=.01 and clearance>=.003)
    if not result['success']:raise RuntimeError('Isolated grasp-pose physical verification failed; no finger action: '+str(result))
    return result
