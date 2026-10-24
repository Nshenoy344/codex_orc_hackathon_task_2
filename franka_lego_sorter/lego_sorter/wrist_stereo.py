"""Measured ChArUco intrinsics/stereo calibration. No robot control imports."""
import cv2
import numpy as np
from .charuco_observation import _core


def detect_corners(image,config):
    board,dictionary=_core().make_board(config)
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    markers,ids,_=cv2.aruco.detectMarkers(gray,dictionary)
    if ids is None:return {},image.copy()
    count,corners,corner_ids=cv2.aruco.interpolateCornersCharuco(markers,ids,gray,board)
    output=image.copy();cv2.aruco.drawDetectedMarkers(output,markers,ids)
    if corner_ids is None:return {},output
    cv2.aruco.drawDetectedCornersCharuco(output,corners,corner_ids)
    return {int(i):p.astype(float)for i,p in zip(corner_ids.ravel(),corners.reshape(-1,2))},output


def transform(rotation,translation):
    result=np.eye(4);result[:3,:3]=rotation;result[:3,3]=np.asarray(translation).reshape(3)
    return result


def solve(samples,config,image_size,realsense_K,realsense_D,tcp_color,T_color_depth=None):
    """Fit webcam lens then fixed-intrinsic stereo; reserve independent views."""
    if len(samples)<8:raise ValueError('Need at least8 diverse paired views, including held-out views')
    board,_=_core().make_board(config)
    points=np.asarray(board.chessboardCorners if not hasattr(board,'getChessboardCorners')else board.getChessboardCorners())
    objects=[];rs_points=[];web_points=[]
    for sample in samples:
        common=sorted(set(map(int,sample['rs_corners']))&set(map(int,sample['webcam_corners'])))
        if len(common)<10:raise ValueError('Each pair needs10 common identified ChArUco corners')
        objects.append(points[common].astype(np.float32))
        rs_points.append(np.array([sample['rs_corners'][str(i)]for i in common],np.float32))
        web_points.append(np.array([sample['webcam_corners'][str(i)]for i in common],np.float32))
    held=[i for i in range(len(samples))if i%4==3];train=[i for i in range(len(samples))if i not in held]
    obj=[objects[i]for i in train];wp=[web_points[i]for i in train];rp=[rs_points[i]for i in train]
    rms,K,D,rvecs,tvecs,intrinsic_std,_,_=cv2.calibrateCameraExtended(obj,wp,tuple(image_size),None,None,flags=cv2.CALIB_FIX_K3)
    # Both intrinsics remain fixed in this independent camera-to-camera fit.
    stereo=cv2.stereoCalibrate(obj,rp,wp,np.asarray(realsense_K,float).copy(),np.asarray(realsense_D,float).copy(),K.copy(),D.copy(),tuple(image_size),flags=cv2.CALIB_FIX_INTRINSIC,criteria=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,100,1e-8))
    srms,_,_,_,_,R,t,_,_=stereo;web_rs=transform(R,t);rs_web=np.linalg.inv(web_rs)
    residuals=[];normals=[]
    for i in range(len(samples)):
        ok,rv,tv=cv2.solvePnP(objects[i],rs_points[i],realsense_K,realsense_D)
        if not ok:raise ValueError('RealSense board PnP failed')
        rb=cv2.Rodrigues(rv)[0];normals.append(rb[:,2]);wb=web_rs@transform(rb,tv)
        projected=cv2.projectPoints(objects[i],cv2.Rodrigues(wb[:3,:3])[0],wb[:3,3],K,D)[0].reshape(-1,2)
        error=np.linalg.norm(projected-web_points[i],axis=1)
        residuals.append(dict(index=i,held_out=i in held,rms_px=float(np.sqrt(np.mean(error**2))),max_px=float(error.max())))
    normals=np.array(normals);max_angle=float(np.degrees(np.arccos(np.clip(normals@normals.T,-1,1))).max())
    normal_singular=np.linalg.svd(normals-normals.mean(axis=0),compute_uv=False)
    focal_relative_std=float(max(intrinsic_std.ravel()[0]/K[0,0],intrinsic_std.ravel()[1]/K[1,1]))
    held_rms=max(r['rms_px']for r in residuals if r['held_out'])
    checks=dict(intrinsic_rms=bool(rms<.8),stereo_rms=bool(srms<1.),held_out_reprojection=bool(held_rms<1.5),board_normal_span=bool(max_angle>=15.),plausible_focal=bool(.2*image_size[0]<K[0,0]<4*image_size[0]and .2*image_size[0]<K[1,1]<4*image_size[0]),plausible_baseline=bool(np.linalg.norm(t)<.5))
    checks.update(two_axis_tilts=bool(normal_singular[1]>=.05),focal_uncertainty=bool(np.isfinite(focal_relative_std)and focal_relative_std<.05))
    result=dict(format='wrist_stereo_calibration_v1',accepted=all(checks.values()),checks=checks,intrinsic_rms_px=float(rms),stereo_rms_px=float(srms),held_out_maximum_view_rms_px=held_rms,board_normal_span_deg=max_angle,view_residuals=residuals,training_indices=train,held_out_indices=held,webcam_camera_matrix=K.tolist(),webcam_distortion=D.ravel().tolist(),image_size=list(image_size),T_logitech_realsense_color=web_rs.tolist(),T_realsense_color_logitech=rs_web.tolist(),T_tcp_logitech=(np.asarray(tcp_color)@rs_web).tolist(),physical_commissioning_verified=False,convention='T_A_B maps column-vector points from B into A; camera frames are optical Xright,Ydown,Zforward',limitations=['TCP composition inherits existing RealSense hand-eye candidate and board scale uncertainty','A webcam pixel is a ray, not a measured depth; intersect a measured surface for metric guidance','Focus, zoom and resolution must remain fixed'])
    if T_color_depth is not None:result['T_realsense_depth_logitech']=(np.linalg.inv(T_color_depth)@rs_web).tolist()
    result.update(normal_excitation_singular_values=normal_singular.tolist(),intrinsic_parameter_standard_deviations=intrinsic_std.ravel().tolist(),maximum_relative_focal_standard_deviation=focal_relative_std)
    return result
