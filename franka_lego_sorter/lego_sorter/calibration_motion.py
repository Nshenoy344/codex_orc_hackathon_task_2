"""Plan-only bounded wrist rotations for measured scene hand-eye acquisition.

This module never executes an arm/gripper action. Parent/orchestrator must review
its saved trajectory and current scene, then use the existing reviewed-plan
executor. Passing numerical checks is not a camera/bracket collision proof.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET
import cv2
import numpy as np
from scipy.interpolate import BPoly
from scipy.spatial.transform import Rotation
from .camera import Frame
from .geometry import matrix
from .scene_registration import register_rgbd


def _angle(rotation):
    return float(np.degrees(np.linalg.norm(cv2.Rodrigues(rotation)[0])))


def rotation_target(start_pose, reference_pose, axis, degrees, max_step_deg=3.1, max_center_drift_m=.005):
    start, reference = matrix(start_pose), matrix(reference_pose)
    if axis not in ('x', 'y'):
        raise ValueError('Calibration rotations are restricted to base X or Y')
    if not np.isfinite(degrees) or abs(degrees) > 15:
        raise ValueError('Absolute reference-relative rotation must be within +/-15 degrees')
    if not np.isfinite(max_step_deg) or not 0 < max_step_deg <= 15:
        raise ValueError('max_step_deg must be in (0,15]')
    if start[2, 3] < .33 or reference[2, 3] < .33:
        raise ValueError('Raise the measured TCP to Z>=0.33m before calibration rotations')
    if not np.isfinite(max_center_drift_m) or not 0 < max_center_drift_m <= .010:
        raise ValueError('max_center_drift_m must be positive and capped at0.010m')
    if np.linalg.norm(start[:3, 3]-reference[:3, 3]) > max_center_drift_m:
        raise ValueError(f'Current TCP has drifted beyond the explicit {max_center_drift_m:.3f}m reference-center acquisition envelope')
    target = start.copy()
    vector = np.zeros(3)
    vector[0 if axis == 'x' else 1] = np.radians(degrees)
    target[:3, :3] = cv2.Rodrigues(vector)[0] @ reference[:3, :3]
    step = _angle(start[:3, :3].T @ target[:3, :3])
    if step > max_step_deg + 1e-6:
        raise ValueError(f'Rotation step {step:.3f}deg exceeds the reviewed {max_step_deg:.3f}deg step bound')
    return target


def _snapshot(path):
    with np.load(path, allow_pickle=False) as data:
        pose = matrix(data['T_base_tcp'])
        for field, bound in [('robot_drift_m', .001), ('robot_drift_rad', .005)]:
            if field not in data or not np.isfinite(float(data[field])) or float(data[field]) > bound:
                raise ValueError(f'{path}: stationary measured capture required ({field})')
        frame = Frame(data['color_bgr'], data['depth_m'], json.loads(str(data['intrinsics_json'])),
                      float(data['captured_monotonic']), json.loads(str(data['metadata_json'])))
    return frame, pose


def scene_guard(reference_frame, current_frame, min_depth_m=.18, board_config=None, max_board_rms_px=.6):
    """Require observed clearance and an actually matched stationary scene.

    It cannot guarantee target-view visibility; acquire and validate another
    snapshot after every executed short step. Identity registration of the first
    snapshot establishes textured coverage only, not hand-eye calibration.
    """
    if not np.isfinite(max_board_rms_px) or not 0 < max_board_rms_px <= .8:
        raise ValueError('max_board_rms_px must be positive and no greater than the explicit0.8px navigation budget')
    if reference_frame.metadata != current_frame.metadata or reference_frame.intrinsics != current_frame.intrinsics:
        raise ValueError('Camera identity/settings/intrinsics changed')
    depth = current_frame.depth_m
    valid = depth[np.isfinite(depth) & (depth > 0)]
    board_quality=None
    if board_config is not None:
        from .charuco_handeye import observe_board
        # A textureless surrounding sheet can have missing stereo depth while
        # the actual ChArUco target is well measured. Qualify that target with
        # PnP AND depth-plane consistency instead of inferring its quality from
        # unrelated background pixels.
        board_quality=dict(reference=observe_board(reference_frame,board_config,max_rms_px=max_board_rms_px),
                           current=observe_board(current_frame,board_config,max_rms_px=max_board_rms_px))
        if len(valid)<1000:
            raise ValueError('Insufficient measured target-region depth')
    elif len(valid) < depth.size*.60:
        raise ValueError('Insufficient valid camera depth to bound the rotation step')
    near = float(np.percentile(valid, 2))
    if near < min_depth_m:
        raise ValueError(f'Observed nearest scene depth {near:.3f}m is below {min_depth_m:.3f}m')
    if board_quality is not None:
        # ChArUco IDs already identify fixed target points. Repeated checker
        # texture can legitimately have several SIFT models; it must not replace
        # the stronger labelled-board pose/depth checks with arbitrary matching.
        t_current_reference=np.array(board_quality['current']['T_color_board']) @ np.linalg.inv(np.array(board_quality['reference']['T_color_board']))
        registration_metrics=dict(method='identified_charuco_board',
            admission_max_rms_px=float(max_board_rms_px),
            measurement_budget_note='Uniform candidate admission only; reprojection residual does not bound pose error. Depth/normal, batch held-out consistency and physical verification remain required.',
            current_image_plane_rms_equivalent_m=float(board_quality['current']['metrics']['reprojection_rms_px']/current_frame.intrinsics['fx']*abs(np.array(board_quality['current']['T_color_board'])[2,3])),
            reference_corner_count=board_quality['reference']['metrics']['corner_count'],
            current_corner_count=board_quality['current']['metrics']['corner_count'],
            T_current_reference=t_current_reference.tolist())
    else:
        registered = register_rgbd(reference_frame, current_frame)
        if registered.metrics['inlier_count'] < 25 or registered.metrics['inlier_fraction'] < .65:
            raise ValueError('Scene overlap is too weak for another calibration rotation')
        registration_metrics=registered.metrics
    return dict(nearest_depth_percentile2_m=near, minimum_required_depth_m=min_depth_m,
                full_image_valid_depth_fraction=float(len(valid)/depth.size),charuco_target_quality=board_quality,
                registration=registration_metrics,
                unmodeled_camera_mount=True,
                warning='Current-view overlap and depth only; inspect a fresh frame after every executed step')



def retreat_target(start_pose, reference_pose, accepted_destination_pose, max_step_deg=3.1):
    """Return only toward a previously accepted raised-center orientation."""
    start, reference, destination = map(matrix, (start_pose, reference_pose, accepted_destination_pose))
    if not np.isfinite(max_step_deg) or not 0 < max_step_deg <= 5.1:
        raise ValueError('Retreat step must remain within the existing5.1degree reviewed navigation bound')
    if min(start[2,3],reference[2,3],destination[2,3]) < .33:
        raise ValueError('Retreat requires all observed TCP heights>=0.33m')
    if max(np.linalg.norm(start[:3,3]-reference[:3,3]),
           np.linalg.norm(destination[:3,3]-reference[:3,3])) > .005:
        raise ValueError('Retreat must stay at the already observed raised rotation center')
    before=_angle(reference[:3,:3].T@start[:3,:3])
    after=_angle(reference[:3,:3].T@destination[:3,:3])
    step=_angle(start[:3,:3].T@destination[:3,:3])
    if before > 15.12 or after > before-.2:
        raise ValueError('Retreat must reduce reference-relative wrist tilt by at least0.2degrees')
    if step > max_step_deg+1e-6:
        raise ValueError(f'Retreat step {step:.3f}degree exceeds {max_step_deg:.3f}degree bound')
    target=start.copy();target[:3,:3]=destination[:3,:3]
    return target


def retreat_scene_guard(reference_frame, current_frame, accepted_destination_frame,
                        board_config, min_depth_m=.18):
    """Qualify navigation back to a good view without accepting a bad sample.

    Calibration admission remains0.600px. The current view may be navigation-
    qualified up to0.700px only for this explicitly decreasing-tilt return; its
    pose is not added to a hand-eye dataset. Both endpoints already accepted for
    calibration use the unchanged strict admission function.
    """
    from .charuco_handeye import observe_board
    for frame in (current_frame,accepted_destination_frame):
        if frame.metadata!=reference_frame.metadata or frame.intrinsics!=reference_frame.intrinsics:
            raise ValueError('Camera identity/settings/intrinsics changed before retreat')
    reference=observe_board(reference_frame,board_config,max_rms_px=.6)
    destination=observe_board(accepted_destination_frame,board_config,max_rms_px=.6)
    current=observe_board(current_frame,board_config,max_rms_px=.7)
    depth=current_frame.depth_m
    valid=depth[np.isfinite(depth)&(depth>0)]
    if len(valid)<1000:
        raise ValueError('Insufficient observed depth for bounded return')
    near=float(np.percentile(valid,2))
    if near<min_depth_m:
        raise ValueError('Observed depth clearance is below bounded-return limit')
    visual=np.array(current['T_color_board'])@np.linalg.inv(np.array(reference['T_color_board']))
    return dict(mode='return_to_previously_accepted_view_only',
                calibration_admission_max_rms_px=.6,navigation_current_max_rms_px=.7,
                current_observation_added_to_calibration=False,
                current_would_pass_calibration_rms=current['metrics']['reprojection_rms_px']<=.6,
                nearest_depth_percentile2_m=near,minimum_required_depth_m=min_depth_m,
                reference_quality=reference['metrics'],destination_quality=destination['metrics'],
                current_navigation_quality=current['metrics'],
                T_current_reference=visual.tolist(),
                warning='Navigation evidence only. Reobserve and pass unchanged calibration admission after retreat.')


def slow_plan(plan, minimum_duration_sec=8.):
    if plan.noop:
        raise ValueError('Requested calibration pose is already reached')
    points = plan.trajectory.joint_trajectory.points
    duration = points[-1].time_from_start.sec+points[-1].time_from_start.nanosec/1e9
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError('Planner returned invalid duration')
    factor = max(1., float(minimum_duration_sec)/duration)
    for point in points:
        seconds=(point.time_from_start.sec+point.time_from_start.nanosec/1e9)*factor
        ns=round(seconds*1e9)
        point.time_from_start.sec=ns//1000000000
        point.time_from_start.nanosec=ns%1000000000
        point.velocities=[v/factor for v in point.velocities]
        point.accelerations=[v/factor**2 for v in point.accelerations]
    return plan


def validate_rotation_plan(data, urdf_path):
    """Dense controller-interpolated joint/FK checks; no hardware imports."""
    root=ET.parse(urdf_path).getroot()
    joints={j.get('name'):j for j in root.findall('joint')}
    bychild={j.find('child').get('link'):j for j in root.findall('joint')}
    chain=[];child=data['tcp_frame']
    while child != data['base_frame']:
        if child not in bychild:raise ValueError('Unknown base-to-TCP URDF chain')
        joint=bychild[child];chain.append(joint);child=joint.find('parent').get('link')
    chain.reverse()
    names=data['joint_names'];points=data['points']
    times=np.array([p['time_sec'] for p in points],float)
    if len(points)<2 or not np.isfinite(times).all() or np.any(np.diff(times)<=0):
        raise ValueError('Invalid timed rotation trajectory')
    for p in points:
        if any(len(p[k])!=len(names) or not np.isfinite(p[k]).all() for k in ('positions','velocities','accelerations')):
            raise ValueError('Malformed rotation trajectory joint arrays')
    spline=BPoly.from_derivatives(times,[[p['positions'],p['velocities'],p['accelerations']] for p in points])
    dense_t=np.unique(np.r_[times,np.linspace(times[0],times[-1],801)])
    dense=spline(dense_t);velocity=spline.derivative()(dense_t)
    def fk(q):
        values=dict(zip(names,q));t=np.eye(4)
        for joint in chain:
            local=np.eye(4);origin=joint.find('origin')
            if origin is not None:
                local[:3,3]=np.fromstring(origin.get('xyz','0 0 0'),sep=' ')
                local[:3,:3]=Rotation.from_euler('xyz',np.fromstring(origin.get('rpy','0 0 0'),sep=' ')).as_matrix()
            if joint.get('type') in ('revolute','continuous'):
                moving=np.eye(4);axis=np.fromstring(joint.find('axis').get('xyz'),sep=' ')
                moving[:3,:3]=Rotation.from_rotvec(values[joint.get('name')]*axis).as_matrix()
                local=local@moving
            t=t@local
        return t
    poses=np.array([fk(q) for q in dense])
    start=matrix(data['T_base_tcp_start']);target=matrix(data['T_base_tcp_target'])
    reference=matrix(data['calibration_rotation']['T_base_tcp_reference'])
    position_drift=np.linalg.norm(poses[:,:3,3]-start[:3,3],axis=1)
    angular_reference=np.array([_angle(reference[:3,:3].T@p[:3,:3]) for p in poses])
    angular_start=np.array([_angle(start[:3,:3].T@p[:3,:3]) for p in poses])
    limit_report={};limits_pass=True
    for index,name in enumerate(names):
        element=joints[name].find('limit')
        lower,upper,limit=map(float,(element.get('lower'),element.get('upper'),element.get('velocity')))
        minimum,maximum=float(dense[:,index].min()),float(dense[:,index].max())
        peak=float(np.max(np.abs(velocity[:,index])))
        passed=minimum>=lower-1e-9 and maximum<=upper+1e-9 and peak<=limit
        limits_pass &= passed
        limit_report[name]=dict(lower=lower,upper=upper,range_rad=[minimum,maximum],
                                peak_speed_rad_s=peak,velocity_limit_rad_s=limit,passed=bool(passed))
    metrics=dict(sample_count=len(dense_t),joint_limits=limit_report,
        duration_sec=float(times[-1]),max_translation_drift_m=float(position_drift.max()),
        start_fk_error_m=float(np.linalg.norm(poses[0,:3,3]-start[:3,3])),
        endpoint_translation_error_m=float(np.linalg.norm(poses[-1,:3,3]-target[:3,3])),
        endpoint_rotation_error_deg=_angle(poses[-1,:3,:3].T@target[:3,:3]),
        maximum_reference_angle_deg=float(angular_reference.max()),
        maximum_step_angle_deg=float(angular_start.max()),minimum_tcp_z_m=float(poses[:,2,3].min()),
        camera_collision_modeled=any('camera' in l.get('name','').lower() and l.find('collision') is not None for l in root.findall('link')),
        limits_of_review='Exact URDF FK and joint limits only; camera/bracket collision geometry is not supplied')
    metrics['passed_numeric_checks']=bool(limits_pass and data['fraction']>=.999999 and
        metrics['duration_sec']>=8. and metrics['max_translation_drift_m']<=.0005 and
        metrics['start_fk_error_m']<=.0005 and metrics['endpoint_translation_error_m']<=.0005 and
        metrics['endpoint_rotation_error_deg']<=.12 and metrics['minimum_tcp_z_m']>=.3295 and
        metrics['maximum_reference_angle_deg']<=15.12 and
        metrics['maximum_step_angle_deg']<=data['calibration_rotation']['max_step_deg']+.12)
    if 'max_center_drift_m' in data['calibration_rotation']:
        center_budget=float(data['calibration_rotation']['max_center_drift_m'])
        if not np.isfinite(center_budget) or not 0 < center_budget <= .010:
            raise ValueError('Invalid explicit reference-center acquisition budget')
        center_drift=float(np.linalg.norm(start[:3,3]-reference[:3,3]))
        if center_drift > center_budget:
            raise ValueError('Measured plan start exceeds its reference-center acquisition budget')
        if not np.allclose(target[:3,3],start[:3,3],atol=1e-5,rtol=0):
            raise ValueError('Center budget must not introduce a commanded TCP translation')
        metrics.update(reference_center_drift_m=center_drift,max_center_drift_m=center_budget)
    if data['calibration_rotation'].get('retreat_to_accepted'):
        retreat=data['calibration_rotation']
        accepted=matrix(retreat['T_base_tcp_accepted_destination'])
        expected=retreat_target(start,reference,accepted,float(retreat['max_step_deg']))
        if not np.allclose(target,expected,atol=1e-8):
            raise ValueError('Retreat target differs from its pinned accepted destination orientation')
        snapshot=Path(retreat['accepted_destination_snapshot'])
        if hashlib.sha256(snapshot.read_bytes()).hexdigest()!=retreat['accepted_destination_sha256']:
            raise ValueError('Accepted destination snapshot changed after retreat planning')
        _,accepted_in_file=_snapshot(snapshot)
        if not np.allclose(accepted,accepted_in_file,atol=1e-8):
            raise ValueError('Retreat destination pose differs from its measured accepted snapshot')
        guard=retreat['scene_guard']
        if guard.get('mode')!='return_to_previously_accepted_view_only' or guard.get('current_observation_added_to_calibration') is not False:
            raise ValueError('Retreat cannot admit current navigation observation to calibration')
    return metrics


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    plan_parser=sub.add_parser('plan',help='Plan and save only; sends no execution goal')
    plan_parser.add_argument('--axis',choices=['x','y'],required=True)
    plan_parser.add_argument('--degrees',type=float,required=True)
    plan_parser.add_argument('--max-step-deg',type=float,default=3.1)
    plan_parser.add_argument('--max-center-drift-m',type=float,default=.005,help='Explicit acquisition envelope only, capped at0.010m; rotation target still holds current XYZ')
    plan_parser.add_argument('--reference',required=True)
    plan_parser.add_argument('--board-config',help='Known ChArUco board for target-local RGBD/PnP qualification')
    plan_parser.add_argument('--max-board-rms-px',type=float,default=.6,help='Explicit candidate admission budget, capped at0.8px; does not relax depth/solver/physical verification')
    plan_parser.add_argument('--current',help='Fresh snapshot; defaults to reference for initial step')
    plan_parser.add_argument('--config',default='config/demo.yaml')
    plan_parser.add_argument('--urdf',required=True)
    plan_parser.add_argument('--output',required=True)
    retreat=sub.add_parser('retreat',help='Plan only: return to a strictly accepted earlier view without admitting current sample')
    retreat.add_argument('--toward',required=True,help='Earlier strictly accepted stationary snapshot')
    retreat.add_argument('--reference',required=True)
    retreat.add_argument('--current',required=True)
    retreat.add_argument('--board-config',required=True)
    retreat.add_argument('--max-step-deg',type=float,default=3.1)
    retreat.add_argument('--config',default='config/demo.yaml')
    retreat.add_argument('--urdf',required=True)
    retreat.add_argument('--output',required=True)
    validate=sub.add_parser('validate',help='Recheck a saved plan offline')
    validate.add_argument('plan');validate.add_argument('--urdf',required=True);validate.add_argument('--output',required=True)
    args=parser.parse_args(argv)
    if args.command=='validate':
        data=json.loads(Path(args.plan).read_text());report=validate_rotation_plan(data,args.urdf)
        report['plan_sha256']=hashlib.sha256(Path(args.plan).read_bytes()).hexdigest()
        Path(args.output).write_text(json.dumps(report,indent=2))
        print(json.dumps(report,indent=2));return 0 if report['passed_numeric_checks'] else 2
    reference_frame,reference_pose=_snapshot(args.reference)
    current_frame,current_pose=_snapshot(args.current or args.reference)
    board_config=None
    if args.board_config:
        import yaml
        from .charuco_handeye import board_configuration
        board_config,_=board_configuration(yaml.safe_load(Path(args.board_config).read_text()))
    if args.command=='retreat':
        destination_frame,destination_pose=_snapshot(args.toward)
        guard=retreat_scene_guard(reference_frame,current_frame,destination_frame,board_config)
    else:
        guard=scene_guard(reference_frame,current_frame,board_config=board_config,max_board_rms_px=args.max_board_rms_px)
    if board_config is not None:
        visual=np.array(guard['T_current_reference'] if args.command=='retreat' else guard['registration']['T_current_reference'])
        robot_angle=_angle(reference_pose[:3,:3].T@current_pose[:3,:3])
        visual_angle=_angle(visual[:3,:3])
        guard['robot_visual_rotation_disagreement_deg']=abs(robot_angle-visual_angle)
        if abs(robot_angle-visual_angle)>1.5:
            raise ValueError('Identified board motion disagrees with measured wrist rotation')
    from .cli import load_config
    from .motion import RosRobot
    from rosidl_runtime_py.convert import message_to_ordereddict
    config=load_config(args.config);robot=RosRobot(config)
    try:
        deadline=time.monotonic()+6
        while True:
            try:start=robot.current_pose();break
            except RuntimeError:
                if time.monotonic()>=deadline:raise
                time.sleep(.05)
        if np.linalg.norm(start[:3,3]-current_pose[:3,3])>.001 or _angle(start[:3,:3].T@current_pose[:3,:3])>.5:
            raise ValueError('Current snapshot no longer matches measured robot pose; capture again')
        if args.command=='retreat':
            target=retreat_target(start,reference_pose,destination_pose,args.max_step_deg)
        else:
            target=rotation_target(start,reference_pose,args.axis,args.degrees,args.max_step_deg,args.max_center_drift_m)
        plan=slow_plan(robot.plan_linear(target))
        data=plan.to_dict();data['trajectory']=message_to_ordereddict(plan.trajectory)
        data.update(base_frame=config['motion']['base_frame'],tcp_frame=config['motion']['tcp_frame'],
            calibration_rotation=dict(axis=getattr(args,'axis','return_to_accepted'),degrees=getattr(args,'degrees',None),max_step_deg=args.max_step_deg,
                T_base_tcp_reference=reference_pose.tolist(),reference_snapshot=args.reference,
                current_snapshot=args.current or args.reference,scene_guard=guard))
        if args.command=='plan':
            data['calibration_rotation'].update(max_center_drift_m=args.max_center_drift_m,
                reference_center_drift_m=float(np.linalg.norm(start[:3,3]-reference_pose[:3,3])),
                center_drift_budget_reason='Explicit raised acquisition envelope only; target XYZ remains the fresh measured TCP position. Actual measured poses remain in hand-eye fitting; controller, depth, path, and solver tolerances are unchanged.')
        if args.command=='retreat':
            data['calibration_rotation'].update(retreat_to_accepted=True,
                T_base_tcp_accepted_destination=destination_pose.tolist(),
                accepted_destination_snapshot=str(Path(args.toward).resolve()),
                accepted_destination_sha256=hashlib.sha256(Path(args.toward).read_bytes()).hexdigest())
        report=validate_rotation_plan(data,args.urdf)
        path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(data,indent=2))
        report['plan_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
        path.with_suffix('.validation.json').write_text(json.dumps(report,indent=2))
        print(json.dumps(report,indent=2));return 0 if report['passed_numeric_checks'] else 2
    finally:robot.close()


if __name__=='__main__':
    raise SystemExit(main())
