"""Immutable-input reuse of a completed noncontact viewing-path review only."""
import hashlib,json
from pathlib import Path


def verify_view_shift_review(plan_path,reviewed,repository_root=None):
    root=Path(repository_root)if repository_root else Path(__file__).resolve().parents[1]
    plan_path=Path(plan_path).resolve();plan=json.loads(plan_path.read_text())
    if 'commissioning_view_shift'not in plan or reviewed.get('purpose')!='noncontact_view_shift' or reviewed.get('passed_numeric_checks')is not True:
        raise ValueError('Only a passing noncontact viewing review may be reused')
    intent=plan['commissioning_view_shift']
    checks=reviewed.get('checks',{})
    required={'application_wrist_branch','stock_hand_cloud_clear','full_arm_cloud_clear','full_arm_table_clear','camera_bracket_assumed_envelope_clear','minimum150mm_tip_clearance','minimum8s_duration','viewing_line','viewing_orientation','viewing_progress'}
    if not required.issubset(checks)or not all(value is True for value in checks.values())or reviewed.get('failed_checks'):
        raise ValueError('Viewing review lacks complete passing path/scene checks')
    for key in ('snapshot','handeye'):
        if Path(intent[key]).resolve()!=Path(reviewed[key]).resolve():raise ValueError('Viewing review evidence differs from planned evidence')
    if abs(float(reviewed['hand_opening_assumed_m'])-float(intent['gripper_width_m']))>1e-9:
        raise ValueError('Reviewed Hand opening differs from planned opening')
    files=[(plan_path,'plan_sha256'),(Path(reviewed['snapshot']),'snapshot_sha256'),
        (Path(reviewed['handeye']),'handeye_sha256'),(Path(reviewed['geometry_report']),'geometry_sha256'),
        (root/'tools/validate_view_shift.py','wrapper_sha256'),(root/'tools/validate_isolated_hover.py','reviewer_sha256'),
        (root/'data/commissioning/fr3_stock_hand.urdf','urdf_sha256')]
    for path,key in files:
        if hashlib.sha256(path.read_bytes()).hexdigest()!=reviewed.get(key):
            raise ValueError('Viewing review input/code changed: '+key)
    return True


def verify_isolated_hover_review(plan_path,reviewed,repository_root=None):
    """Reuse only the identical fully reviewed noncontact200mm hover inputs."""
    root=Path(repository_root)if repository_root else Path(__file__).resolve().parents[1]
    path=Path(plan_path).resolve();plan=json.loads(path.read_text());intent=plan.get('isolated_observation',{})
    if intent.get('purpose')!='noncontact_hover200' or intent.get('planning_method')not in('cartesian_plan_only','joint_space_plan_only')or reviewed.get('passed_numeric_checks')is not True:
        raise ValueError('Only a passing isolated200mm noncontact hover review may be reused')
    checks=reviewed.get('checks',{})
    required={'application_wrist_branch','joint_position_and_speed_limits','complete_planning_fraction','minimum12s_duration','start_fk_consistent','endpoint_fk_consistent','endpoint_rotation_consistent','minimum150mm_tip_clearance','stock_hand_cloud_clear','full_arm_cloud_clear','full_arm_table_clear','camera_bracket_assumed_envelope_clear'}
    if not required.issubset(checks)or not all(value is True for value in checks.values())or reviewed.get('failed_checks'):
        raise ValueError('Isolated hover review lacks complete passing path/scene checks')
    for key in('snapshot','geometry_report'):
        if Path(intent[key]).resolve()!=Path(reviewed[key]).resolve():raise ValueError('Isolated hover evidence differs from planned evidence')
    if intent['snapshot_sha256']!=reviewed['snapshot_sha256']:raise ValueError('Isolated hover planned snapshot changed')
    geometry=json.loads(Path(reviewed['geometry_report']).read_text())
    if Path(geometry['handeye_artifact']).resolve()!=Path(reviewed['handeye']).resolve():raise ValueError('Isolated hover handeye differs from measured geometry')
    if abs(float(reviewed['hand_opening_assumed_m'])-float(intent['gripper_width_m']))>1e-9:raise ValueError('Reviewed hover Hand opening differs from planned opening')
    if intent['planning_method']=='cartesian_plan_only'and float(reviewed['max_cartesian_line_deviation_m'])>.0005:raise ValueError('Reviewed Cartesian hover is not a straight segment')
    files=[(path,'plan_sha256'),(Path(reviewed['snapshot']),'snapshot_sha256'),(Path(reviewed['handeye']),'handeye_sha256'),
        (Path(reviewed['geometry_report']),'geometry_sha256'),(root/'tools/validate_isolated_hover.py','reviewer_sha256'),
        (root/'data/commissioning/fr3_stock_hand.urdf','urdf_sha256')]
    for filename,key in files:
        if hashlib.sha256(filename.read_bytes()).hexdigest()!=reviewed.get(key):raise ValueError('Isolated hover review input/code changed: '+key)
    return True


def verify_isolated_approach_review(plan_path,reviewed,repository_root=None):
    root=Path(repository_root)if repository_root else Path(__file__).resolve().parents[1]
    path=Path(plan_path);plan=json.loads(path.read_text());meta=plan.get('isolated_diagnostic_stage',{})
    checks=reviewed.get('checks',{});required={'application_wrist_branch','approach20mm_nominal_tip_clearance','approach_straight_line','approach_progress','stock_hand_cloud_clear','full_arm_cloud_clear','full_arm_table_clear','camera_bracket_assumed_envelope_clear','minimum12s_duration'}
    if meta.get('stage')!='approach30'or reviewed.get('purpose')!='isolated_approach30'or reviewed.get('passed_numeric_checks')is not True or not required.issubset(checks)or not all(v is True for v in checks.values())or reviewed.get('failed_checks'):
        raise ValueError('Only the complete passing noncontact approach30 review is supported')
    for key in('snapshot','handeye'):
        if Path(meta[key]).resolve()!=Path(reviewed[key]).resolve():raise ValueError('Approach evidence differs from planned evidence')
    if Path(meta['targets']).resolve()!=Path(reviewed['source_targets']).resolve()or abs(float(meta['gripper_width_m'])-float(reviewed['hand_opening_assumed_m']))>1e-9:raise ValueError('Approach targets/opening differ from reviewed inputs')
    files=[(path,'plan_sha256'),(Path(reviewed['snapshot']),'snapshot_sha256'),(Path(reviewed['handeye']),'handeye_sha256'),(Path(reviewed['geometry_report']),'geometry_sha256'),(Path(reviewed['source_targets']),'source_targets_sha256'),(root/'tools/validate_isolated_stage.py','wrapper_sha256'),(root/'tools/validate_isolated_hover.py','reviewer_sha256'),(root/'data/commissioning/fr3_stock_hand.urdf','urdf_sha256')]
    for filename,key in files:
        if hashlib.sha256(filename.read_bytes()).hexdigest()!=reviewed.get(key):raise ValueError('Approach review input/code changed: '+key)
    return True
