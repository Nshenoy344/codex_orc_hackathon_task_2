"""Offline known-board observations and measured-pose hand-eye candidates.

The board geometry/metric scale is fixed once, never re-estimated per view.
Default scale provenance is the first RGB-D estimate, not a claimed physical
measurement. Numerical acceptance never establishes physical picking accuracy.
"""
import argparse
import json
from pathlib import Path
import sys
import cv2
import numpy as np
import yaml
from .camera import Frame
from .charuco_observation import _core, _camera, fit_depth_plane, plane_points
from .geometry import matrix
from .scene_handeye import solve_scene_handeye, CalibrationRejected


def board_configuration(value):
    """Accept a plain board config or the existing first-view inspection artifact."""
    if 'candidate_board_config' in value:
        config=value['candidate_board_config']
        candidate=(value.get('candidates') or [{}])[0]
        evidence=dict(first_view_depth_square_scale=candidate.get('depth_square_scale'),
                      first_view_depth_plane=candidate.get('depth_plane'))
    else:
        config=value.get('board_config',value)
        evidence={}
    keys=('dictionary','squares_x','squares_y','square_length_m','marker_length_m','legacy_pattern')
    config={k:config.get(k,False) if k=='legacy_pattern' else config[k] for k in keys}
    _core().make_board(config)
    return config,dict(evidence,scale_source='aligned_depth_estimate',
        physical_scale_verified=False,
        scale_uncertainty_note='First-view corner-pair MAD quantifies internal fit spread, not systematic depth-scale error; no printed dimension is asserted exact')


def _metric_rigid_pose(object_points, observed_points):
    a,b=np.asarray(object_points,float),np.asarray(observed_points,float)
    if a.shape!=b.shape or a.ndim!=2 or a.shape[1]!=3 or len(a)<12 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Need at least12 finite identified metric corner pairs')
    ac,bc=a-a.mean(axis=0),b-b.mean(axis=0)
    if np.linalg.matrix_rank(ac,tol=1e-7)<2 or np.linalg.matrix_rank(bc,tol=1e-7)<2:
        raise ValueError('Metric corners are collinear')
    u,_,vt=np.linalg.svd(ac.T@bc)
    rotation=vt.T@np.diag([1.,1.,np.linalg.det(vt.T@u.T)])@u.T
    pose=np.eye(4);pose[:3,:3]=rotation;pose[:3,3]=b.mean(axis=0)-rotation@a.mean(axis=0)
    errors=np.linalg.norm(a@rotation.T+pose[:3,3]-b,axis=1)
    residual=dict(corner_count=len(a),metric_rms_m=float(np.sqrt(np.mean(errors**2))),
                  metric_max_m=float(errors.max()),metric_median_m=float(np.median(errors)),
                  metric_rms_limit_m=.0015,metric_max_limit_m=.003)
    if residual['metric_rms_m']>.0015 or residual['metric_max_m']>.003:
        raise ValueError('Fixed-scale in-plane metric corner fit failed: '+json.dumps(residual))
    return pose,residual


def _identified_corners(frame,board,dictionary,k,d):
    gray=cv2.cvtColor(frame.color_bgr,cv2.COLOR_BGR2GRAY)
    if hasattr(cv2.aruco,'CharucoDetector'):
        params=cv2.aruco.CharucoParameters();params.cameraMatrix=k;params.distCoeffs=d
        corners,ids,_,_=cv2.aruco.CharucoDetector(board,params).detectBoard(gray)
    else:
        markers,marker_ids,_=cv2.aruco.detectMarkers(gray,dictionary)
        if marker_ids is None:raise ValueError('No identified board markers')
        _,corners,ids=cv2.aruco.interpolateCornersCharuco(markers,marker_ids,gray,board,cameraMatrix=k,distCoeffs=d)
    if ids is None or len(ids)<12:raise ValueError('Too few identified ChArUco corners for RGBD plane pose')
    points=(board.getChessboardCorners() if hasattr(board,'getChessboardCorners') else board.chessboardCorners)[ids.ravel()]
    return points,corners.reshape(-1,2),ids.ravel()


def observe_board(frame, fixed_board_config, min_corners=12, max_rms_px=.6,
                  require_depth_consistency=True, estimator='pnp'):
    """Estimate T_color_board with an unchanged dictionary/layout/metric scale."""
    if estimator not in ('pnp','rgbd_plane'):raise ValueError('Unknown board pose estimator')
    if estimator=='rgbd_plane' and not require_depth_consistency:
        raise ValueError('RGBD plane estimator requires qualified depth-plane consistency')
    core=_core();board,dictionary=core.make_board(fixed_board_config)
    k,d=_camera(frame)
    rv,tv,rms,count=core.detect(frame.color_bgr,k,d,board,dictionary,int(min_corners))
    if not np.isfinite(rms) or rms>max_rms_px:
        raise ValueError(f'ChArUco reprojection RMS {rms:.3f}px exceeds {max_rms_px:.3f}px')
    pose=np.eye(4);pose[:3,:3]=cv2.Rodrigues(rv)[0];pose[:3,3]=tv.ravel()
    matrix(pose,'T_color_board')
    metrics=dict(corner_count=int(count),reprojection_rms_px=float(rms),
                 max_reprojection_rms_px=float(max_rms_px))
    x,y=fixed_board_config['squares_x'],fixed_board_config['squares_y']
    square=fixed_board_config['square_length_m']
    outer=np.array([[0,0,0],[x*square,0,0],[x*square,y*square,0],[0,y*square,0]],np.float32)
    polygon=cv2.projectPoints(outer,rv,tv,k,d)[0].reshape(-1,2)
    try:
        normal,offset,plane=fit_depth_plane(frame,polygon)
        angle=float(np.degrees(np.arccos(np.clip(abs(normal@pose[:3,2]),0,1))))
        object_points=(board.getChessboardCorners() if hasattr(board,'getChessboardCorners') else board.chessboardCorners)
        pixels=cv2.projectPoints(object_points,rv,tv,k,d)[0].reshape(-1,2)
        h,w=frame.depth_m.shape
        visible=(pixels[:,0]>=1)&(pixels[:,1]>=1)&(pixels[:,0]<w-1)&(pixels[:,1]<h-1)
        if visible.sum()<4:raise ValueError('Too few projected board corners for depth-plane comparison')
        observed=plane_points(frame,pixels[visible],normal,offset)
        predicted=object_points[visible]@pose[:3,:3].T+pose[:3,3]
        dz=np.abs(predicted[:,2]-observed[:,2])
        consistency=dict(plane=plane,normal_disagreement_deg=angle,
            compared_visible_corner_count=int(visible.sum()),median_depth_residual_m=float(np.median(dz)),
            maximum_depth_residual_m=float(dz.max()),rms_depth_residual_m=float(np.sqrt(np.mean(dz**2))))
        consistency['passed']=bool(angle<=3. and np.median(dz)<=.004 and dz.max()<=.008)
        metrics['depth_consistency']=consistency
        if not consistency['passed'] and require_depth_consistency:
            raise ValueError('PnP/depth board-plane consistency failed: '+json.dumps(consistency))
    except ValueError as exc:
        if require_depth_consistency:raise
        metrics['depth_consistency']=dict(passed=False,reason=str(exc),required=False)
    pnp_pose=pose.copy()
    if estimator=='rgbd_plane':
        points,measured_pixels,corner_ids=_identified_corners(frame,board,dictionary,k,d)
        if len(points)<min_corners:raise ValueError('Too few metric corner pairs for configured minimum')
        observed_xyz=plane_points(frame,measured_pixels,normal,offset)
        pose,metric_fit=_metric_rigid_pose(points,observed_xyz)
        if np.any((points@pose[:3,:3].T+pose[:3,3])[:,2]<=0):
            raise ValueError('RGBD fitted board is behind the camera')
        normal_difference=float(np.degrees(np.arccos(np.clip(pnp_pose[:3,2]@pose[:3,2],-1,1))))
        if normal_difference>3.:
            raise ValueError('RGBD/PnP board normals disagree beyond the unchanged3deg bound')
        fit_pixels=cv2.projectPoints(points,cv2.Rodrigues(pose[:3,:3])[0],pose[:3,3],k,d)[0].reshape(-1,2)
        metric_fit.update(normal_disagreement_from_pnp_deg=normal_difference,
            rgbd_reprojection_rms_px=float(np.sqrt(np.mean(np.sum((fit_pixels-measured_pixels)**2,axis=1)))),
            identified_corner_ids=corner_ids.tolist(),depth_plane_rms_m=plane['rms_m'],
            estimator_note='Detected identified corner rays intersect independently fitted depth plane; rigid Kabsch fit uses unchanged metric board geometry. PnP admission/depth consistency and batch validation remain required.')
        metrics['rgbd_metric_fit']=metric_fit
    metrics['pose_estimator']=estimator
    return dict(T_color_board=pose.tolist(),T_color_board_pnp=pnp_pose.tolist(),metrics=metrics,
                fixed_board_config=dict(fixed_board_config),
                physical_metric_scale_verified=False)


def load_observation(path):
    with np.load(path,allow_pickle=False) as data:
        for key,bound in [('robot_drift_m',.001),('robot_drift_rad',.005)]:
            if key not in data or not np.isfinite(float(data[key])) or float(data[key])>bound:
                raise ValueError(f'{path}: missing/excessive stationary robot {key}')
        pose=matrix(data['T_base_tcp'],'T_base_tcp')
        frame=Frame(data['color_bgr'],data['depth_m'],json.loads(str(data['intrinsics_json'])),
                    float(data['captured_monotonic']),json.loads(str(data['metadata_json'])))
        extrinsic=matrix(data['T_color_depth'],'T_color_depth') if 'T_color_depth' in data else None
        return frame,pose,extrinsic


def inspect_observations(paths, fixed_board_config, min_corners=12,max_rms_px=.6,
                         require_depth_consistency=True,estimator='pnp'):
    paths=[Path(p).expanduser().resolve() for p in paths]
    if not paths or len(set(paths))!=len(paths):
        raise CalibrationRejected('Provide distinct measured observation paths')
    reports=[];robot=[];target=[];extrinsics=[];metadata=None;intrinsics=None
    for path in paths:
        try:
            frame,g,extrinsic=load_observation(path)
            if metadata is None:metadata,intrinsics=frame.metadata,frame.intrinsics
            if frame.metadata!=metadata or frame.intrinsics!=intrinsics:
                raise ValueError('Camera identity/settings/intrinsics differ from the first observation')
            observation=observe_board(frame,fixed_board_config,min_corners,max_rms_px,require_depth_consistency,estimator)
            reports.append(dict(path=str(path),accepted=True,T_base_tcp=g.tolist(),**observation))
            robot.append(g);target.append(observation['T_color_board'])
            if extrinsic is not None:extrinsics.append(extrinsic)
        except (OSError,KeyError,ValueError,cv2.error) as exc:
            reports.append(dict(path=str(path),accepted=False,reason=str(exc)))
    if any(not r['accepted'] for r in reports):
        raise CalibrationRejected('One or more fixed-board observations failed; no pose is silently discarded',
                                  dict(observations=reports,sample_count=len(paths)))
    if extrinsics and not np.allclose(extrinsics,extrinsics[0],atol=1e-7):
        raise CalibrationRejected('RealSense factory color/depth extrinsics changed',dict(observations=reports))
    return dict(observations=reports,robot=robot,target=target,camera_metadata=metadata,
                camera_intrinsics=intrinsics,T_color_depth=None if not extrinsics else extrinsics[0],
                factory_extrinsics_observation_count=len(extrinsics))


def calibrate_charuco(paths,board_config,solver_config=None,base_frame='fr3_link0',tcp_frame='fr3_hand_tcp',
                      min_corners=12,max_rms_px=.6,require_depth_consistency=True,estimator='pnp'):
    config,scale=board_configuration(board_config)
    inspected=inspect_observations(paths,config,min_corners,max_rms_px,require_depth_consistency,estimator)
    evidence={k:inspected[k] for k in ('observations','camera_metadata','camera_intrinsics','factory_extrinsics_observation_count')}
    try:
        result=solve_scene_handeye(inspected['robot'],inspected['target'],solver_config)
    except CalibrationRejected as exc:
        exc.diagnostics.update(evidence,board_config=config,scale_provenance=scale,pose_estimator=estimator)
        raise
    result['parameters']=dict(base_frame=base_frame,tcp_frame=tcp_frame,**config)
    result['pose_estimator']=estimator
    result['scale_source']=scale['scale_source']
    result['physical_scale_verified']=False
    result['physical_commissioning_verified']=False
    result['scale_provenance']=scale
    result['T_base_board']=result.pop('T_base_reference')
    result['diagnostics'].update(evidence)
    extrinsic=inspected['T_color_depth']
    if extrinsic is not None:
        tcp_depth=np.array(result['T_tcp_color'])@extrinsic
        result.update(T_color_depth=extrinsic.tolist(),T_tcp_depth=tcp_depth.tolist(),
                      T_depth_tcp=np.linalg.inv(tcp_depth).tolist())
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('observations',nargs='+');p.add_argument('--board-config',required=True)
    p.add_argument('--output',required=True);p.add_argument('--solver-config')
    p.add_argument('--base-frame',default='fr3_link0');p.add_argument('--tcp-frame',default='fr3_hand_tcp')
    p.add_argument('--min-corners',type=int,default=12);p.add_argument('--max-rms-px',type=float,default=.6)
    p.add_argument('--inspect-only',action='store_true')
    p.add_argument('--estimator',choices=['pnp','rgbd_plane'],default='pnp')
    a=p.parse_args(argv)
    config_value=yaml.safe_load(Path(a.board_config).read_text())
    solver=yaml.safe_load(Path(a.solver_config).read_text()) if a.solver_config else None
    if solver:solver=solver.get('scene_handeye',solver)
    try:
        if a.inspect_only:
            config,scale=board_configuration(config_value)
            inspected=inspect_observations(a.observations,config,a.min_corners,a.max_rms_px,estimator=a.estimator)
            result={k:v for k,v in inspected.items() if k not in ('robot','target','T_color_depth')}
            result.update(board_config=config,scale_provenance=scale,accepted_observations=True,
                          calibration_accepted=False,physical_commissioning_verified=False)
        else:
            result=calibrate_charuco(a.observations,config_value,solver,a.base_frame,a.tcp_frame,a.min_corners,a.max_rms_px,estimator=a.estimator)
        code=0
    except (CalibrationRejected,ValueError,OSError,KeyError,cv2.error) as exc:
        result=dict(diagnostics=exc.diagnostics if isinstance(exc,CalibrationRejected) else
                    dict(accepted=False,reason=str(exc),physical_commissioning_verified=False))
        code=2
    path=Path(a.output);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(yaml.safe_dump(result,sort_keys=False));temp.replace(path)
    summary={k:v for k,v in result.items() if k not in ('diagnostics','observations')}
    if 'diagnostics' in result:
        summary['diagnostics']={k:v for k,v in result['diagnostics'].items() if k!='observations'}
    print(json.dumps(summary,indent=2));return code


if __name__=='__main__':sys.exit(main())
