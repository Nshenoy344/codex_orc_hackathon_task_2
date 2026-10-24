"""Narrow reviewed held-brick approach; no opening or production bookkeeping."""
import hashlib,json
from pathlib import Path
from .held_view_checkpoint import verify_checkpoint

def verify_review(plan_path,reviewed):
    root=Path(__file__).resolve().parents[1]
    path=Path(plan_path);plan=json.loads(path.read_text());m=plan['isolated_diagnostic_stage']
    required={'application_wrist_branch','minimum15s_duration','stock_hand_cloud_clear','full_arm_cloud_clear','full_arm_table_clear','camera_bracket_assumed_envelope_clear','recorded_environment_boxes_clear','held_box20mm_above_global_observed_template_max','held_box_observed_cloud_clear','held_box_static_input_boxes_clear','approach_cartesian_line','approach_bounded_progress'}
    checks=reviewed.get('checks',{})
    if m.get('stage')!='first_loose_place_approach' or m.get('gripper_action_authorized')is not False or reviewed.get('purpose')!='isolated_first_loose_place_approach' or reviewed.get('passed_numeric_checks')is not True or not required.issubset(checks)or not all(v is True for v in checks.values())or reviewed.get('failed_checks')or reviewed.get('camera_mode')!='oriented_contact':
        raise ValueError('Complete passing exact loose-placement approach review required')
    files=[(path,'plan_sha256'),(root/'tools/validate_first_loose_place_approach.py','wrapper_sha256'),(root/'tools/validate_isolated_hover.py','reviewer_sha256'),(root/'lego_sorter/contact_camera_envelope.py','camera_helper_sha256'),(root/'data/commissioning/fr3_stock_hand.urdf','urdf_sha256'),(Path(reviewed['geometry_report']),'geometry_sha256')]
    files.append((root/'lego_sorter/environment_boxes.py','environment_helper_sha256'))
    for key in ('checkpoint','placement_target','scene_snapshot','handeye','environment_boxes'):
        if Path(m[key]).resolve()!=Path(reviewed[key]).resolve():raise ValueError('Approach evidence path changed: '+key)
        files.append((Path(m[key]),key+'_sha256'))
    for filename,key in files:
        if hashlib.sha256(filename.read_bytes()).hexdigest()!=reviewed.get(key):raise ValueError('Approach evidence/code changed: '+key)
    verify_checkpoint(m['checkpoint'])
    for item in json.loads(Path(m['placement_target']).read_text())['source_files']:
        if hashlib.sha256(Path(item['path']).read_bytes()).hexdigest()!=item['sha256']:raise ValueError('Placement target source changed')
    if abs(float(m['gripper_width_m'])-float(reviewed['hand_opening_assumed_m']))>1e-9:raise ValueError('Reviewed opening differs')

def execute_approach(robot,plan,meta):
    before_hand=robot.gripper_state();verify_checkpoint(meta['checkpoint'],robot.current_pose(),float(before_hand['width_m']))
    actual=robot.execute_linear_plan(plan);after_hand=robot.gripper_state()
    if abs(float(after_hand['width_m'])-float(before_hand['width_m']))>.0002:raise RuntimeError('Hand holding width changed during approach; inspect before continuation')
    return dict(measured_after=actual.tolist(),gripper_before=before_hand,gripper_after=after_hand,holding_width_verified=True,arm_action_sent=True,release_action_sent=False,production_count_changed=False)
