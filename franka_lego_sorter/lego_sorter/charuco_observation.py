"""Offline measured ChArUco board inspection; no hardware or calibration claims.

Dictionary superset identification cannot distinguish dictionary sizes when only
shared low marker IDs are printed. Automatic board dimensions require a complete
sequential marker set, complete ChArUco corner coverage, and a visible white
outer margin. RGB-D scale is an estimate to verify against a printed measurement.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import cv2
import numpy as np
from .camera import Frame

FAMILIES=('DICT_4X4_1000','DICT_5X5_1000','DICT_6X6_1000','DICT_7X7_1000',
          'DICT_ARUCO_ORIGINAL','DICT_APRILTAG_16h5','DICT_APRILTAG_25h9',
          'DICT_APRILTAG_36h10','DICT_APRILTAG_36h11')


def _core():
    path=Path(__file__).resolve().parents[2]/'franka_realsense_handeye_ros2/franka_realsense_handeye/board.py'
    spec=importlib.util.spec_from_file_location('_existing_charuco_board',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def _camera(frame):
    p=frame.intrinsics
    return np.array([[p['fx'],0,p['ppx']],[0,p['fy'],p['ppy']],[0,0,1.]],float),np.array(p.get('coeffs',[0]*5),float)


def scan_dictionaries(image):
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    result=[]
    for family in FAMILIES:
        if not hasattr(cv2.aruco,family):continue
        dictionary=cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco,family))
        corners,ids,_=cv2.aruco.detectMarkers(gray,dictionary)
        if ids is not None:
            result.append(dict(dictionary=family,marker_ids=ids.ravel().tolist(),
                               marker_corners_px=[c.reshape(4,2).tolist() for c in corners],
                               marker_count=len(ids)))
    return sorted(result,key=lambda x:x['marker_count'],reverse=True)


def fit_depth_plane(frame,polygon):
    mask=np.zeros(frame.depth_m.shape,np.uint8)
    cv2.fillConvexPoly(mask,np.round(polygon).astype(np.int32),255)
    mask=cv2.erode(mask,np.ones((7,7),np.uint8))
    ys,xs=np.where((mask>0)&np.isfinite(frame.depth_m)&(frame.depth_m>.1)&(frame.depth_m<2.))
    if len(xs)<200:raise ValueError('Too little reliable aligned board depth')
    if len(xs)>5000:
        selected=np.linspace(0,len(xs)-1,5000).astype(int);xs,ys=xs[selected],ys[selected]
    points=np.array([frame.deproject(float(x),float(y),float(frame.depth_m[y,x])) for x,y in zip(xs,ys)])
    keep=np.ones(len(points),bool)
    for _ in range(5):
        center=points[keep].mean(axis=0)
        _,_,vt=np.linalg.svd(points[keep]-center,full_matrices=False)
        normal=vt[-1];offset=-normal@center
        errors=np.abs(points@normal+offset)
        threshold=max(.0015,float(np.percentile(errors[keep],85)))
        keep=errors<=min(.006,threshold)
        if keep.sum()<200:raise ValueError('Board depth is not a supported plane')
    rms=float(np.sqrt(np.mean(errors[keep]**2)))
    if rms>.0025:raise ValueError(f'Board plane RMS too high: {rms:.4f}m')
    return normal,offset,dict(rms_m=rms,inlier_count=int(keep.sum()),sample_count=len(points))


def plane_points(frame,pixels,normal,offset):
    rays=np.array([frame.deproject(float(u),float(v),1.) for u,v in pixels])
    denominators=rays@normal
    if np.any(np.abs(denominators)<1e-5):raise ValueError('Board plane ray is degenerate')
    scale=-offset/denominators
    if np.any(scale<=0):raise ValueError('Board plane intersects behind camera')
    return rays*scale[:,None]


def estimate_square_scale(frame,pixels,object_points,normal,offset):
    points=plane_points(frame,pixels,normal,offset)
    estimates=[]
    for i in range(len(points)):
        for j in range(i):
            units=np.linalg.norm(object_points[i]-object_points[j])
            if units>=2.:
                estimates.append(float(np.linalg.norm(points[i]-points[j])/units))
    if len(estimates)<20:raise ValueError('Insufficient corner span for physical square scale')
    value=float(np.median(estimates))
    mad=float(np.median(np.abs(np.array(estimates)-value)))
    if not .005<value<.10 or mad/value>.03:
        raise ValueError('Depth-derived square scale is invalid or inconsistent')
    return value,dict(pair_count=len(estimates),median_m=value,mad_m=mad,relative_mad=mad/value)


def _margin_evidence(gray,polygon):
    center=polygon.mean(axis=0)
    white=float(np.percentile(gray,80));black=float(np.percentile(gray,10));cut=(white+black)/2
    scores=[]
    for index in range(4):
        a,b=polygon[index],polygon[(index+1)%4]
        edge=b-a;normal=np.array([-edge[1],edge[0]])
        normal/=np.linalg.norm(normal)
        if normal@((a+b)/2-center)<0:normal=-normal
        # Quarter-square wide strips outside each complete checkerboard edge.
        spacing=np.linalg.norm(edge)/7.
        inner1=a+.12*edge+normal*2;inner2=b-.12*edge+normal*2
        outer1=inner1+normal*max(4.,spacing*.20);outer2=inner2+normal*max(4.,spacing*.20)
        vertices=np.array([inner1,inner2,outer2,outer1])
        h,w=gray.shape
        if np.any(vertices[:,0]<0) or np.any(vertices[:,1]<0) or np.any(vertices[:,0]>=w) or np.any(vertices[:,1]>=h):
            scores.append(0.);continue
        mask=np.zeros(gray.shape,np.uint8);cv2.fillConvexPoly(mask,np.round(vertices).astype(np.int32),255)
        scores.append(float(np.mean(gray[mask>0]>cut)))
    return scores


def inspect_charuco(frame,board_config=None):
    gray=cv2.cvtColor(frame.color_bgr,cv2.COLOR_BGR2GRAY)
    k,d=_camera(frame);core=_core()
    scans=scan_dictionaries(frame.color_bgr)
    result=dict(dictionary_scan=scans,accepted_observation=False,physical_calibration_verified=False,
                exact_dictionary_size_identifiable=False,candidates=[],rejections=[])
    if not scans:
        result['reason']='No supported marker dictionary detected';return result
    best_count=scans[0]['marker_count']
    if len(scans)>1 and scans[1]['marker_count']>=max(4,best_count*.5):
        result['reason']='Competing marker dictionary families';return result
    scan=scans[0];ids=np.array(scan['marker_ids'],np.int32).reshape(-1,1)
    markers=[np.array(c,np.float32).reshape(1,4,2) for c in scan['marker_corners_px']]
    dictionary=cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco,scan['dictionary']))
    dimensions=[]
    if board_config:
        dimensions.append((board_config['squares_x'],board_config['squares_y'],'supplied_geometry'))
        result['supplied_dictionary_matches_family']=board_config['dictionary']==scan['dictionary']
    elif sorted(scan['marker_ids'])==list(range(best_count)) and best_count>=8:
        for x in range(3,17):
            for y in range(3,17):
                if (x*y)//2==best_count:
                    dimensions.append((x,y,'complete_marker_set_hypothesis'))
    if not dimensions:
        result['reason']='Board dimensions unavailable; a partial/sequential marker set cannot establish geometry'
        return result
    for x,y,source in dimensions:
        try:
            p=dict(dictionary=scan['dictionary'],squares_x=x,squares_y=y,
                   square_length_m=1.,marker_length_m=.75,legacy_pattern=False)
            board,_=core.make_board(p)
            count,corners,corner_ids=cv2.aruco.interpolateCornersCharuco(markers,ids,gray,board,cameraMatrix=k,distCoeffs=d)
            if corner_ids is None or count<max(8,.8*(x-1)*(y-1)):
                raise ValueError('Incomplete ChArUco corner coverage')
            points=(board.getChessboardCorners() if hasattr(board,'getChessboardCorners') else board.chessboardCorners)[corner_ids.ravel()]
            ok,rv,tv=cv2.solvePnP(points,corners,k,d)
            if not ok:raise ValueError('PnP failed')
            projected,_=cv2.projectPoints(points,rv,tv,k,d)
            rms=float(np.sqrt(np.mean(np.sum((projected.reshape(-1,2)-corners.reshape(-1,2))**2,axis=1))))
            if not np.isfinite(rms) or rms>1.:
                raise ValueError(f'Geometry reprojection RMS {rms:.2f}px exceeds1px')
            outer=np.array([[0,0,0],[x,0,0],[x,y,0],[0,y,0]],np.float32)
            polygon=cv2.projectPoints(outer,rv,tv,k,d)[0].reshape(-1,2)
            margins=_margin_evidence(gray,polygon)
            if not board_config and min(margins)<.85:
                raise ValueError('Full outer white margin not observed; board may be larger or clipped')
            normal,offset,plane=fit_depth_plane(frame,polygon)
            square,scale=estimate_square_scale(frame,corners.reshape(-1,2),points,normal,offset)
            marker_xyz=plane_points(frame,np.array(scan['marker_corners_px']).reshape(-1,2),normal,offset).reshape(-1,4,3)
            marker_edges=np.linalg.norm(np.roll(marker_xyz,-1,axis=1)-marker_xyz,axis=2)
            marker_length=float(np.median(marker_edges));ratio=marker_length/square
            if not .5<ratio<.95:raise ValueError('Measured marker/square ratio implausible')
            metric=dict(dictionary=scan['dictionary'],squares_x=x,squares_y=y,
                        square_length_m=square,marker_length_m=marker_length,legacy_pattern=False)
            metric_board,_=core.make_board(metric)
            rv2,tv2,rms2,n=core.detect(frame.color_bgr,k,d,metric_board,dictionary,8)
            transform=np.eye(4);transform[:3,:3]=cv2.Rodrigues(rv2)[0];transform[:3,3]=tv2.ravel()
            supplied_scale=None
            if board_config:
                supplied_scale=dict(square_length_m=board_config['square_length_m'],
                    depth_estimate_ratio=square/board_config['square_length_m'],
                    scale_agrees=abs(square/board_config['square_length_m']-1)<.03)
            result['candidates'].append(dict(board_config=metric,geometry_source=source,
                marker_ids=scan['marker_ids'],charuco_corner_ids=corner_ids.ravel().tolist(),
                charuco_corners_px=corners.reshape(-1,2).tolist(),board_polygon_px=polygon.tolist(),
                corner_count=int(n),reprojection_rms_px=float(rms2),depth_plane=plane,
                depth_square_scale=scale,marker_to_square_ratio=ratio,white_margin_fractions=margins,
                supplied_scale=supplied_scale,T_color_board_depth_scaled=transform.tolist(),
                metric_scale_verified_by_physical_measurement=False))
        except (ValueError,cv2.error) as exc:
            result['rejections'].append(dict(squares_x=x,squares_y=y,reason=str(exc)))
    result['candidates'].sort(key=lambda c:c['reprojection_rms_px'])
    if len(result['candidates'])!=1:
        result['reason']='No unique complete board geometry passed the image/depth checks'
        return result
    result['accepted_observation']=True
    result['reason']='Unique visible board geometry with low PnP residual; metric scale remains RGB-D estimated until independently measured'
    result['candidate_board_config']=result['candidates'][0]['board_config']
    return result


def annotate(frame,result):
    image=frame.color_bgr.copy()
    for candidate in result['candidates']:
        cv2.polylines(image,[np.round(candidate['board_polygon_px']).astype(np.int32)],True,(0,255,0),2)
        for point,index in zip(candidate['charuco_corners_px'],candidate['charuco_corner_ids']):
            p=tuple(np.round(point).astype(int));cv2.circle(image,p,3,(0,255,255),-1)
            cv2.putText(image,str(index),p,cv2.FONT_HERSHEY_SIMPLEX,.3,(0,0,255),1)
    return image


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('snapshot');p.add_argument('--output',required=True)
    p.add_argument('--board-config',help='Optional explicit board config JSON; geometry is not assumed if omitted')
    a=p.parse_args();data=np.load(a.snapshot,allow_pickle=False)
    frame=Frame(data['color_bgr'],data['depth_m'],json.loads(str(data['intrinsics_json'])),float(data['captured_monotonic']),json.loads(str(data['metadata_json'])))
    result=inspect_charuco(frame,json.loads(Path(a.board_config).read_text()) if a.board_config else None)
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.with_suffix('.json').write_text(json.dumps(result,indent=2))
    cv2.imwrite(str(out.with_suffix('.png')),annotate(frame,result))
    print(json.dumps({k:v for k,v in result.items() if k not in ('dictionary_scan','candidates')},indent=2))
    return 0 if result['accepted_observation'] else 2


if __name__=='__main__':raise SystemExit(main())
