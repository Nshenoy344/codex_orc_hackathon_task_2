"""Execute exactly one reviewed10mm lift while preserving an existing Hand grasp."""
import hashlib,json
from pathlib import Path
import numpy as np
from .geometry import matrix


def verify_review(plan_path,reviewed,repository_root=None):
    root=Path(repository_root)if repository_root else Path(__file__).resolve().parents[1];path=Path(plan_path);p=json.loads(path.read_text());m=p.get('isolated_diagnostic_stage',{})
    required={'application_wrist_branch','minimum12s_duration','stock_hand_cloud_clear','full_arm_cloud_clear','full_arm_table_clear','camera_bracket_assumed_envelope_clear','held_box_no_new_obstacle','held_box_initial_nonpenetration_with_uncertainty','held_box_monotonic_floor_clearance','held_box_final_floor_gain','lift_straight_line','lift_progress'};checks=reviewed.get('checks',{})
    if m.get('stage')=='carry_observe300':
        required.remove('held_box_initial_nonpenetration_with_uncertainty');required.update({'held_box_carry_positive_floor_clearance','minimum15s_duration'})
    if m.get('stage')=='carry_template_view':
        required.difference_update({'held_box_initial_nonpenetration_with_uncertainty','held_box_monotonic_floor_clearance','held_box_final_floor_gain'})
        required.update({'minimum15s_duration','held_box_view_floor_clearance','template_view_tcp_height','held_box_recorded_environment_clear','recorded_environment_boxes_clear'})
    if m.get('stage')not in('lift10','carry_observe300','carry_template_view')or reviewed.get('purpose')!='isolated_'+m['stage']or reviewed.get('passed_numeric_checks')is not True or not required.issubset(checks)or not all(v is True for v in checks.values())or reviewed.get('failed_checks')or reviewed.get('camera_mode')!='oriented_contact':raise ValueError('Only complete passing isolated lift10 review is supported')
    files=[(path,'plan_sha256'),(root/'tools/validate_isolated_hover.py','reviewer_sha256'),(root/'tools/validate_isolated_lift10.py','wrapper_sha256'),(root/'lego_sorter/contact_camera_envelope.py','camera_helper_sha256'),(root/'data/commissioning/fr3_stock_hand.urdf','urdf_sha256'),(Path(reviewed['geometry_report']),'geometry_sha256')]
    extra=('view_hint','carry_result','source_proof','scene_snapshot','environment_boxes')if m['stage']=='carry_template_view'else(('lift10_result',)if m['stage']=='carry_observe300'else())
    if m['stage']=='carry_template_view':files.append((root/'lego_sorter/environment_boxes.py','environment_helper_sha256'))
    for key in('targets','hold_result','snapshot','handeye')+extra:
        if Path(m[key]).resolve()!=Path(reviewed[key]).resolve():raise ValueError('Lift evidence path changed: '+key)
        files.append((Path(reviewed[key]),key+'_sha256'))
    if abs(float(m['gripper_width_m'])-float(reviewed['hand_opening_assumed_m']))>1e-9:raise ValueError('Reviewed lift Hand opening changed')
    for filename,key in files:
        if hashlib.sha256(filename.read_bytes()).hexdigest()!=reviewed.get(key):raise ValueError('Lift review input/code changed: '+key)
    return True


def execute_lift10(robot,plan,meta):
    if meta.get('stage')not in('lift10','carry_observe300','carry_template_view'):raise ValueError('Only isolated upward lift/carry stage is supported')
    hold=json.loads(Path(meta['hold_result']).read_text())
    if hold.get('success')is not True or hold.get('hand_grasp_and_body_width_verified')is not True:raise ValueError('Verified existing Hand hold required')
    expected=float(hold['gripper_after']['width_m']);before_hand=robot.gripper_state()
    if abs(float(before_hand['width_m'])-expected)>.0002:raise RuntimeError('Holding width changed before lift; no arm command sent')
    before=matrix(robot.current_pose());actual=matrix(robot.execute_linear_plan(plan));after_hand=robot.gripper_state()
    up=float(actual[2,3]-before[2,3]);width_ok=abs(float(after_hand['width_m'])-expected)<=.0002
    required_up=.005 if meta['stage']=='lift10'else float(plan.target_pose[2,3]-plan.start_pose[2,3])-.003
    result=dict(measured_before=before.tolist(),measured_after=actual.tolist(),measured_base_z_increase_m=up,gripper_before=before_hand,gripper_after=after_hand,
        holding_width_verified=bool(width_ok),success=bool(up>=required_up and width_ok),gripper_action_sent=False,release_action_sent=False,production_count_changed=False,brick_lifted_visually_verified=False)
    if not result['success']:raise RuntimeError('Test lift mechanical evidence failed; keep pending and inspect: '+str(result))
    return result
