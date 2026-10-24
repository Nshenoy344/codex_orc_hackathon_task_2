"""Measured commissioning artifacts; capture is read-only, motion is explicit."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import cv2
import numpy as np
from .cli import load_config
from .camera import RealSenseCamera
from .motion import RosRobot, LinearPlan


def save_observation(robot, camera, output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    start = robot.current_pose()
    frame = camera.capture_burst()
    finish = robot.current_pose()
    drift = float(np.linalg.norm(start[:3, 3]-finish[:3, 3]))
    angle = float(np.arccos(np.clip((np.trace(start[:3, :3].T @ finish[:3, :3])-1)/2, -1, 1)))
    if drift > .001 or angle > .005:
        raise RuntimeError(f'Robot moved during capture: {drift:.6f}m, {angle:.6f}rad')
    extrinsics=({'T_color_depth':camera.T_color_depth} if hasattr(camera,'T_color_depth') else {})
    np.savez_compressed(output, color_bgr=frame.color_bgr, depth_m=frame.depth_m,
                        intrinsics_json=json.dumps(frame.intrinsics),
                        metadata_json=json.dumps(frame.metadata),
                        captured_monotonic=frame.captured_monotonic,
                        T_base_tcp=finish, robot_drift_m=drift, robot_drift_rad=angle,**extrinsics)
    if not cv2.imwrite(str(output.with_suffix('.png')), frame.color_bgr):
        raise RuntimeError('Cannot save observation image')
    report=dict(snapshot=str(output), T_base_tcp=finish.tolist(), robot_drift_m=drift,
                robot_drift_rad=angle, camera=frame.metadata)
    output.with_suffix('.json').write_text(json.dumps(report, indent=2))
    return report



def source_view_target(start_pose, distance_m, source_reference):
    """Bounded raised-X return toward an actually recorded source observation."""
    from .geometry import matrix
    start=matrix(start_pose)
    distance=float(distance_m)
    if not np.isfinite(distance) or not -.20<=distance<0:
        raise ValueError('Source-view return requires a negativeX step of at most20cm')
    if not (.20<=start[0,3]<=.60 and -.15<=start[1,3]<=.15 and .33<=start[2,3]<=.40):
        raise ValueError('Current TCP is outside the observed raised traversal corridor')
    path=Path(source_reference).expanduser().resolve()
    with np.load(path,allow_pickle=False) as observation:
        source=matrix(observation['T_base_tcp'])
        for key,bound in [('robot_drift_m',.001),('robot_drift_rad',.005)]:
            if key not in observation or not np.isfinite(float(observation[key])) or float(observation[key])>bound:
                raise ValueError('Source intent must be a stationary measured robot observation')
    if not (.20<=source[0,3]<=.60 and -.15<=source[1,3]<=.15 and .33<=source[2,3]<=.40):
        raise ValueError('Source reference is not an observed raised source pose')
    if abs(source[2,3]-start[2,3])>.02 or abs(source[1,3]-start[1,3])>.03:
        raise ValueError('Source reference does not belong to this raisedX traversal corridor')
    target=start.copy();target[0,3]+=distance
    if not .20<=target[0,3]<=.60:
        raise ValueError('Source-view target leaves the observed X corridor')
    if (target[0,3]<source[0,3]-.005 or
            abs(target[0,3]-source[0,3])>=abs(start[0,3]-source[0,3])):
        raise ValueError('Return must approach observed sourceX without passing it by more than5mm')
    intent=dict(purpose='return_to_observed_source',axis='x',distance_m=distance,
        minimum_tcp_z_m=.33,source_reference_snapshot=str(path),
        source_reference_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        T_base_tcp_source_reference=source.tolist(),
        source_reference_role='Pinned observed sourceX intent; currentY/Z/orientation are held throughout return')
    return target,intent


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['record','plan-lift','plan-correction','plan-board-view','plan-source-view','execute-plan'])
    parser.add_argument('--config', default='config/demo.yaml')
    parser.add_argument('--output', required=True)
    parser.add_argument('--distance-m',type=float,default=.20,help='Board-view positiveX or source-view negativeX shift, magnitude<=0.20m')
    parser.add_argument('--source-reference',help='Measured raised source observation NPZ required for negativeX return')
    parser.add_argument('--plan-file', help='Reviewed plan JSON for execute-plan')
    args=parser.parse_args()
    c=load_config(args.config)
    robot=RosRobot(c)
    try:
        # Wait for discovery without sending an action goal.
        deadline=time.monotonic()+6
        while True:
            try:
                start=robot.current_pose()
                break
            except RuntimeError:
                if time.monotonic()>=deadline:raise
        if args.action=='plan-lift':
            target=start.copy()
            target[2,3]+=.020
            if not (.20<start[0,3]<.35 and -.15<start[1,3]<.15 and .20<start[2,3]<.40):
                raise RuntimeError('Start pose outside the observed commissioning region')
            plan=robot.plan_linear(target)
            if plan.noop:raise RuntimeError('Lift unexpectedly produced no motion')
            # Slow this first diagnostic retreat to at least eight seconds.
            jt=plan.trajectory.joint_trajectory
            duration=jt.points[-1].time_from_start.sec+jt.points[-1].time_from_start.nanosec/1e9
            factor=max(1.,8./duration)
            for point in jt.points:
                seconds=(point.time_from_start.sec+point.time_from_start.nanosec/1e9)*factor
                ns=round(seconds*1e9)
                point.time_from_start.sec=ns//1000000000
                point.time_from_start.nanosec=ns%1000000000
                point.velocities=[v/factor for v in point.velocities]
                point.accelerations=[a/factor**2 for a in point.accelerations]
            from rosidl_runtime_py.convert import message_to_ordereddict
            data=plan.to_dict()
            data['trajectory']=message_to_ordereddict(plan.trajectory)
            data['base_frame']=c['motion']['base_frame']
            data['tcp_frame']=c['motion']['tcp_frame']
            Path(args.output).parent.mkdir(parents=True,exist_ok=True)
            Path(args.output).write_text(json.dumps(data,indent=2))
            print(json.dumps({k:v for k,v in data.items() if k not in ('trajectory','points')},indent=2))
            return
        if args.action in ('plan-board-view','plan-source-view'):
            if not (.20<start[0,3]<.60 and -.15<start[1,3]<.15 and .33<=start[2,3]<=.40):
                raise RuntimeError('Current pose is outside the observed board-view commissioning region')
            distance=float(args.distance_m)
            if args.action=='plan-source-view':
                if not args.source_reference:parser.error('plan-source-view requires --source-reference')
                target,intent=source_view_target(start,distance,args.source_reference)
            else:
                if not np.isfinite(distance) or not 0<distance<=.20 or start[0,3]+distance>.60:
                    raise ValueError('Board-view shift must stay in the observed raised workspace and be <=20cm')
                target=start.copy();target[0,3]+=distance
                intent=dict(axis='x',distance_m=distance,
                    user_instruction='20cm away from base toward top of RealSense image',minimum_tcp_z_m=.33)
            from .calibration_motion import slow_plan
            from rosidl_runtime_py.convert import message_to_ordereddict
            plan=slow_plan(robot.plan_linear(target),minimum_duration_sec=max(8.,100.*abs(distance)))
            data=plan.to_dict();data['trajectory']=message_to_ordereddict(plan.trajectory)
            data.update(base_frame=c['motion']['base_frame'],tcp_frame=c['motion']['tcp_frame'],
                        commissioning_translation=intent)
            Path(args.output).write_text(json.dumps(data,indent=2))
            print(json.dumps({k:v for k,v in data.items() if k not in ('trajectory','points')},indent=2))
            return
        if args.action=='plan-correction':
            if not args.plan_file:parser.error('plan-correction requires the original --plan-file')
            original=json.loads(Path(args.plan_file).read_text())
            if original.get('endpoint_correction') is not None:
                raise RuntimeError('Chained automatic corrections are not permitted')
            from .endpoint_servo import plan_endpoint_correction
            from .calibration_motion import slow_plan
            from rosidl_runtime_py.convert import message_to_ordereddict
            correction=plan_endpoint_correction(robot,np.array(original['T_base_tcp_target']))
            if correction is None:raise RuntimeError('The original goal is already within strict tolerance')
            slow_plan(correction.linear_plan)
            data=correction.linear_plan.to_dict()
            data['endpoint_correction']=correction.to_dict()
            data['trajectory']=message_to_ordereddict(correction.linear_plan.trajectory)
            data['base_frame']=c['motion']['base_frame'];data['tcp_frame']=c['motion']['tcp_frame']
            data['original_plan_file']=args.plan_file
            Path(args.output).write_text(json.dumps(data,indent=2))
            print(json.dumps({k:v for k,v in data['endpoint_correction'].items() if k!='linear_plan'},indent=2))
            return
        if args.action=='execute-plan':
            if not args.plan_file:parser.error('execute-plan requires --plan-file')
            data=json.loads(Path(args.plan_file).read_text())
            if any(data[k]!=c['motion'][k] for k in ('base_frame','tcp_frame')):
                raise RuntimeError('Plan frames differ from configured robot')
            before=np.array(data['T_base_tcp_start']);target=np.array(data['T_base_tcp_target'])
            if 'endpoint_correction' in data:
                validation_path=Path(args.plan_file).with_suffix('.validation.json')
                reviewed=json.loads(validation_path.read_text())
                if (reviewed.get('plan_sha256')!=hashlib.sha256(Path(args.plan_file).read_bytes()).hexdigest() or
                        reviewed.get('passed_numeric_checks') is not True):
                    raise RuntimeError('Correction requires its unchanged passing independent review')
                import importlib.util
                module_path=Path(__file__).resolve().parents[1]/'tools'/'validate_endpoint_correction.py'
                import sys
                sys.path.insert(0,str(module_path.parent))
                spec=importlib.util.spec_from_file_location('_validate_endpoint',module_path)
                validator=importlib.util.module_from_spec(spec);spec.loader.exec_module(validator)
                if not validator.review(args.plan_file,'data/commissioning/fr3_stock_hand.urdf')['passed_numeric_checks']:
                    raise RuntimeError('Correction no longer passes independent numeric checks')
            elif 'commissioning_translation' in data:
                validation_path=Path(args.plan_file).with_suffix('.validation.json')
                reviewed=json.loads(validation_path.read_text())
                if (reviewed.get('plan_sha256')!=hashlib.sha256(Path(args.plan_file).read_bytes()).hexdigest() or
                        reviewed.get('passed_numeric_checks') is not True):
                    raise RuntimeError('Translation requires its unchanged passing independent review')
                import importlib.util
                module_path=Path(__file__).resolve().parents[1]/'tools'/'validate_translation_plan.py'
                import sys
                sys.path.insert(0,str(module_path.parent))
                spec=importlib.util.spec_from_file_location('_validate_translation',module_path)
                validator=importlib.util.module_from_spec(spec);spec.loader.exec_module(validator)
                if not validator.validate(args.plan_file,'data/commissioning/fr3_stock_hand.urdf')['passed_numeric_checks']:
                    raise RuntimeError('Translation no longer passes independent numeric checks')
            elif 'commissioning_view_shift' in data:
                intent=data['commissioning_view_shift']
                reviewed=json.loads(Path(args.plan_file).with_suffix('.validation.json').read_text())
                from .review_cache import verify_view_shift_review
                try:verify_view_shift_review(args.plan_file,reviewed)
                except ValueError as exc:raise RuntimeError(str(exc)) from exc
                opening=float(robot.gripper_state()['width_m'])
                if abs(opening-float(intent['gripper_width_m']))>.0002:
                    raise RuntimeError('Measured Hand opening changed since viewing plan')
                # Reuse the identical reviewed cloud calculation only after every
                # evidence/code/model hash matches. Shared execution below still
                # checks fresh TCP/joints and dense application wrist bounds.
            elif 'isolated_diagnostic_stage' in data:
                meta=data['isolated_diagnostic_stage']
                reviewed=json.loads(Path(args.plan_file).with_suffix('.validation.json').read_text())
                if meta.get('stage')in ('object_approach30','object_grasp_pose'):
                    from .object_diagnostic import verify_review
                    try:verify_review(args.plan_file,reviewed)
                    except ValueError as exc:raise RuntimeError(str(exc)) from exc
                elif meta.get('stage')=='postrelease_retreat350':
                    from .postrelease_retreat import verify_review
                    try:verify_review(args.plan_file,reviewed)
                    except ValueError as exc:raise RuntimeError(str(exc)) from exc
                elif meta.get('stage')=='first_loose_release_pose':
                    from .loose_release_pose import verify_review
                    try:verify_review(args.plan_file,reviewed)
                    except ValueError as exc:raise RuntimeError(str(exc)) from exc
                elif meta.get('stage')=='first_loose_place_approach':
                    from .loose_place_approach import verify_review
                    try:verify_review(args.plan_file,reviewed)
                    except ValueError as exc:raise RuntimeError(str(exc)) from exc
                elif meta.get('stage')in ('lift10','carry_observe300','carry_template_view'):
                    from .isolated_lift10 import verify_review
                    try:verify_review(args.plan_file,reviewed)
                    except ValueError as exc:raise RuntimeError(str(exc)) from exc
                elif meta.get('stage')=='grasp_pose':
                    from .isolated_grasp_pose import verify_review
                    try:verify_review(args.plan_file,reviewed)
                    except ValueError as exc:raise RuntimeError(str(exc)) from exc
                else:
                    from .review_cache import verify_isolated_approach_review
                    try:verify_isolated_approach_review(args.plan_file,reviewed)
                    except ValueError as exc:raise RuntimeError(str(exc)) from exc
                opening=float(robot.gripper_state()['width_m'])
                if abs(opening-float(meta['gripper_width_m']))>.0002:
                    raise RuntimeError('Measured Hand opening changed since isolated stage planning')
            elif 'isolated_observation' in data:
                observation=data['isolated_observation']
                if observation.get('purpose')!='noncontact_hover200' or observation.get('planning_method')not in ('joint_space_plan_only','cartesian_plan_only'):
                    raise RuntimeError('Only the reviewed noncontact observation stage is supported')
                if target[2,3]<.15 or not np.allclose(target[:3,2],[0,0,-1],atol=.002):
                    raise RuntimeError('Observation target must remain raised with vertical tool')
                validation_path=Path(args.plan_file).with_suffix('.validation.json')
                reviewed=json.loads(validation_path.read_text())
                from .review_cache import verify_isolated_hover_review
                try:verify_isolated_hover_review(args.plan_file,reviewed)
                except ValueError as exc:raise RuntimeError(str(exc)) from exc
                geometry=json.loads(Path(reviewed['geometry_report']).read_text())
                hover=np.asarray(geometry['poses']['grasp'],float).copy();hover[2,3]+=.200
                if np.linalg.norm(target[:3,3]-hover[:3,3])>.0005:
                    raise RuntimeError('Observation target does not match measured brick200mm hover')
                opening=float(robot.gripper_state()['width_m'])
                if abs(opening-float(observation['gripper_width_m']))>.0002:
                    raise RuntimeError('Measured Hand opening changed since observation planning')
                # Identical evidence/core/model files retain their complete review;
                # fresh pose/joints/opening and application branch checks remain.
            elif 'calibration_rotation' in data:
                from .calibration_motion import validate_rotation_plan
                validation_path=Path(args.plan_file).with_suffix('.validation.json')
                reviewed=json.loads(validation_path.read_text())
                digest=hashlib.sha256(Path(args.plan_file).read_bytes()).hexdigest()
                if reviewed.get('plan_sha256')!=digest or reviewed.get('passed_numeric_checks') is not True:
                    raise RuntimeError('Rotation plan is changed or lacks a passing hash-pinned review')
                report=validate_rotation_plan(data,'data/commissioning/fr3_stock_hand.urdf')
                if report.get('passed_numeric_checks') is not True:
                    raise RuntimeError('Rotation trajectory no longer passes independent kinematic checks')
            elif (not np.allclose(target[:3,:3],before[:3,:3],atol=1e-8) or
                  not np.allclose(target[:3,3]-before[:3,3],[0,0,.020],atol=1e-8)):
                raise RuntimeError('Expected the bounded +20mm retreat or a reviewed calibration rotation')
            from moveit_msgs.msg import RobotTrajectory
            from rosidl_runtime_py.set_message import set_message_fields
            trajectory=RobotTrajectory()
            set_message_fields(trajectory,data['trajectory'])
            plan=LinearPlan(before,target,tuple(data['start_joint_names']),
                            tuple(data['start_joint_positions']),trajectory,data['fraction'])
            try:
                if 'endpoint_correction' in data:
                    from .endpoint_servo import EndpointCorrection,execute_endpoint_correction
                    ec=data['endpoint_correction']
                    correction=EndpointCorrection(plan,np.array(ec['physical_goal']),np.array(ec['measured_before']),
                        np.array(ec['command_target']),float(ec['strict_position_tolerance_m']),float(ec['strict_rotation_tolerance_rad']))
                    correction_result=execute_endpoint_correction(robot,correction)
                    actual=np.array(correction_result['measured_after'])
                elif data.get('isolated_diagnostic_stage',{}).get('stage')in ('object_approach30','object_grasp_pose'):
                    from .object_diagnostic import execute_stage
                    object_diagnostic_result=execute_stage(robot,plan,data['isolated_diagnostic_stage'])
                    actual=np.array(object_diagnostic_result['measured_after'])
                elif data.get('isolated_diagnostic_stage',{}).get('stage')=='postrelease_retreat350':
                    from .postrelease_retreat import execute_retreat
                    retreat_result=execute_retreat(robot,plan,data['isolated_diagnostic_stage'])
                    actual=np.array(retreat_result['measured_after'])
                elif data.get('isolated_diagnostic_stage',{}).get('stage')=='first_loose_release_pose':
                    from .loose_release_pose import execute_release_pose
                    release_pose_result=execute_release_pose(robot,plan,data['isolated_diagnostic_stage'])
                    actual=np.array(release_pose_result['measured_after'])
                elif data.get('isolated_diagnostic_stage',{}).get('stage')=='first_loose_place_approach':
                    from .loose_place_approach import execute_approach
                    approach_result=execute_approach(robot,plan,data['isolated_diagnostic_stage'])
                    actual=np.array(approach_result['measured_after'])
                elif data.get('isolated_diagnostic_stage',{}).get('stage')in ('lift10','carry_observe300','carry_template_view'):
                    from .isolated_lift10 import execute_lift10
                    lift10_result=execute_lift10(robot,plan,data['isolated_diagnostic_stage'])
                    actual=np.array(lift10_result['measured_after'])
                elif data.get('isolated_diagnostic_stage',{}).get('stage')=='grasp_pose':
                    from .isolated_grasp_pose import execute_grasp_pose
                    grasp_pose_result=execute_grasp_pose(robot,plan,data['isolated_diagnostic_stage'])
                    actual=np.array(grasp_pose_result['measured_after'])
                else:
                    actual=robot.execute_linear_plan(plan)
            except BaseException as exc:
                failure=dict(success=False,status='execution_or_verification_failed',
                             reason=str(exc),plan_file=args.plan_file,
                             physical_motion_may_have_occurred=True,
                             T_base_tcp_before=before.tolist(),T_base_tcp_target=target.tolist())
                try:
                    actual=robot.current_pose()
                    failure['T_base_tcp_after']=actual.tolist()
                    failure['target_position_error_m']=float(np.linalg.norm(actual[:3,3]-target[:3,3]))
                except Exception as state_error:
                    failure['state_error']=str(state_error)
                Path(args.output).write_text(json.dumps(failure,indent=2))
                raise
            report=dict(success=True,T_base_tcp_before=before.tolist(),T_base_tcp_after=actual.tolist(),
                        measured_delta_m=(actual[:3,3]-before[:3,3]).tolist(),plan_file=args.plan_file)
            if 'endpoint_correction' in data:report['endpoint_correction_result']=correction_result
            if data.get('isolated_diagnostic_stage',{}).get('stage')in ('object_approach30','object_grasp_pose'):report['object_diagnostic_result']=object_diagnostic_result
            if data.get('isolated_diagnostic_stage',{}).get('stage')=='postrelease_retreat350':report['postrelease_retreat_result']=retreat_result
            if data.get('isolated_diagnostic_stage',{}).get('stage')=='first_loose_release_pose':report['first_loose_release_pose_result']=release_pose_result
            if data.get('isolated_diagnostic_stage',{}).get('stage')=='first_loose_place_approach':report['loose_place_approach_result']=approach_result
            if data.get('isolated_diagnostic_stage',{}).get('stage')=='grasp_pose':report['isolated_grasp_pose_result']=grasp_pose_result
            if data.get('isolated_diagnostic_stage',{}).get('stage')in ('lift10','carry_observe300','carry_template_view'):report['isolated_lift10_result']=lift10_result
            Path(args.output).write_text(json.dumps(report,indent=2))
            print(json.dumps(report,indent=2))
            return
        with RealSenseCamera(c['camera']) as camera:
            # Give TF/state discovery a bounded opportunity before first capture.
            deadline=time.monotonic()+5
            while True:
                try:
                    robot.current_pose()
                    break
                except RuntimeError:
                    if time.monotonic()>=deadline:raise
            print(json.dumps(save_observation(robot,camera,args.output),indent=2))
    finally:robot.close()


if __name__=='__main__':main()
