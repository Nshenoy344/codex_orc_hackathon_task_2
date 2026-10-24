"""Measured continuation state after an unsuccessful viewing endpoint verification.

A checkpoint never changes the failed motion result and is not motion success.
It permits downstream planners to use actual geometry, retaining fresh-start
checks and existing holding evidence rather than pretending the old goal is met.
"""
import hashlib,json,time
from pathlib import Path
import numpy as np
from .geometry import matrix
from .endpoint_servo import pose_error
from .isolated_brick import load_frame


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_checkpoint(snapshot_path,result_path,actual_pose,gripper):
    snapshot_path=Path(snapshot_path).resolve();result_path=Path(result_path).resolve();result=json.loads(result_path.read_text())
    # This exact failure occurs after the execution action succeeded in the shared
    # adapter; a timeout/reflex/unknown action outcome is deliberately unsupported.
    if result.get('success')is not False or not result.get('reason','').startswith('Pose verification failed:')or result.get('status')!='execution_or_verification_failed':raise ValueError('Requires preserved endpoint-verification failure, not unknown execution/reflex outcome')
    plan_path=Path(result['plan_file']).resolve();plan=json.loads(plan_path.read_text());meta=plan['isolated_diagnostic_stage']
    if meta.get('stage')!='carry_template_view':raise ValueError('Only the reviewed held template-view attempt is supported')
    hold=json.loads(Path(meta['hold_result']).read_text());proof=json.loads(Path(meta['source_proof']).read_text())
    if hold.get('success')is not True or hold.get('hand_grasp_and_body_width_verified')is not True or proof.get('combined_pick_evidence')is not True:raise ValueError('Verified Hand hold and cleared original source are required')
    actual=matrix(actual_pose);_,captured=load_frame(snapshot_path);after=matrix(result['T_base_tcp_after']);width=float(gripper['width_m'])
    d,a=pose_error(captured,actual)
    if np.linalg.norm(d)>.001 or a>.01:raise ValueError('Fresh TCP does not match actual capture within1mm/.01rad')
    d,a=pose_error(after,actual)
    if np.linalg.norm(d)>.003 or a>.01:raise ValueError('Fresh measured holding state is outside3mm/.01rad acquisition envelope')
    if not np.isfinite(width)or abs(width-.0158)>.001 or abs(width-float(hold['gripper_after']['width_m']))>.0002:raise ValueError('Fresh opening does not preserve verified brick-body hold')
    goal=matrix(plan['T_base_tcp_target']);delta,angle=pose_error(goal,actual)
    out=dict(format='measured_held_view_checkpoint_v1',created_monotonic=time.monotonic(),T_base_tcp=actual.tolist(),capture_T_base_tcp=captured.tolist(),width_m=width,
        gripper_state=gripper,holding_width_verified=True,original_source_cleared=True,motion_success_claimed=False,associated_execution_result_success=False,
        associated_target_position_error_m=float(np.linalg.norm(delta)),associated_target_rotation_error_rad=float(angle),associated_target_error_translation_m=delta.tolist(),
        arm_action_sent=False,hand_action_sent=False,checkpoint_semantics='Fresh measured held-brick state and actual SDK observation after a failed viewing endpoint check. This is not acceptance of the old goal or permission to replay it.')
    paths={'snapshot':snapshot_path,'associated_result':result_path,'associated_plan':plan_path}
    for key in('hold_result','source_proof','handeye','targets','environment_boxes'):paths[key]=Path(meta[key]).resolve()
    for key,path in paths.items():out[key]=str(path);out[key+'_sha256']=digest(path)
    return out


def verify_checkpoint(path,actual_pose=None,width_m=None):
    p=Path(path);c=json.loads(p.read_text())
    if c.get('format')!='measured_held_view_checkpoint_v1'or c.get('motion_success_claimed')is not False or c.get('associated_execution_result_success')is not False or c.get('holding_width_verified')is not True:raise ValueError('Invalid measured held-view checkpoint semantics')
    for key in('snapshot','associated_result','associated_plan','hold_result','source_proof','handeye','targets','environment_boxes'):
        if digest(c[key])!=c[key+'_sha256']:raise ValueError('Held-view checkpoint evidence changed: '+key)
    stored=matrix(c['T_base_tcp'])
    if actual_pose is not None:
        delta,angle=pose_error(stored,matrix(actual_pose))
        if np.linalg.norm(delta)>.001 or angle>.01:raise ValueError('Fresh motion start differs from measured checkpoint by more than1mm/.01rad')
    if width_m is not None and(not np.isfinite(width_m)or abs(float(width_m)-float(c['width_m']))>.0002):raise ValueError('Fresh holding opening changed from checkpoint')
    return c
