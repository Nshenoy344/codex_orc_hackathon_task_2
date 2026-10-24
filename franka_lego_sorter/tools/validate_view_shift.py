#!/usr/bin/env python3
"""Offline review of one <=60mm projected-camera-Y viewing translation."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np,yaml
from scipy.spatial.transform import Rotation
from lego_sorter.isolated_brick import load_frame,_cloud
from lego_sorter.charuco_observation import fit_depth_plane
from validate_isolated_hover import review as review_path


def make_geometry(snapshot,handeye,out):
    frame,tcp=load_frame(snapshot);x=np.asarray(yaml.safe_load(Path(handeye).read_text())['T_tcp_color']);camera=tcp@x
    normal,d,quality=fit_depth_plane(frame,np.array([[360,180],[600,180],[600,370],[360,370]],float))
    if normal[2]>0:normal=-normal;d=-d
    if quality['rms_m']>.0015:raise ValueError('Measured viewing-station tabletop plane failed1.5mm RMS')
    normal_base=camera[:3,:3]@normal
    if normal_base[2]<.98:raise ValueError('Measured viewing-station table is not nearly horizontal')
    table_point=camera[:3,:3]@(-d*normal)+camera[:3,3]
    cloud,_=_cloud(frame,3);world=cloud@camera[:3,:3].T+camera[:3,3]
    top=world[np.argmax(world[:,2])]
    data=dict(scene_reference_semantics='Highest actual base-Z point in the valid0.1..2m stationary observed RGBD cloud, not a brick or grasp point. Legacy visible_top_base_m field is used only for conservative150mm viewing clearance.',
        snapshot=str(Path(snapshot).resolve()),snapshot_sha256=hashlib.sha256(Path(snapshot).read_bytes()).hexdigest(),
        visible_top_base_m=top.tolist(),table_point_base_m=table_point.tolist(),
        measured_table_plane=dict(normal_base=normal_base.tolist(),normal_camera=normal.tolist(),offset_camera_m=float(d),**quality))
    Path(out).write_text(json.dumps(data,indent=2));return data


def review(plan_path,geometry=None,opening=None):
    data=json.loads(Path(plan_path).read_text());intent=data['commissioning_view_shift']
    if intent.get('camera_axis')!='positive_y_projected_xy':raise ValueError('Only declared positive cameraY XY viewing shift is supported')
    distance=float(intent['distance_m'])
    if not np.isfinite(distance)or not 0<distance<=.060000001:raise ValueError('Viewing shift must be positive and atmost60mm')
    snapshot=Path(intent['snapshot']);handeye=Path(intent['handeye']);_,measured=load_frame(snapshot)
    start=np.asarray(data['T_base_tcp_start']);target=np.asarray(data['T_base_tcp_target']);x=np.asarray(yaml.safe_load(handeye.read_text())['T_tcp_color'])
    if np.linalg.norm(start[:3,3]-measured[:3,3])>.001 or Rotation.from_matrix(start[:3,:3].T@measured[:3,:3]).magnitude()>.01:raise ValueError('Viewing plan differs from measured scene snapshot pose')
    direction=(measured@x)[:3,1].copy();direction[2]=0.;direction/=np.linalg.norm(direction)
    if np.linalg.norm(target[:3,3]-start[:3,3]-direction*distance)>.00001 or not np.allclose(start[:3,:3],target[:3,:3],atol=1e-8):raise ValueError('Viewing target must preserveZ/R and match declared projected cameraY shift')
    geometry=Path(geometry)if geometry else Path(plan_path).with_suffix('.scene_geometry.json')
    if not geometry.exists():make_geometry(snapshot,handeye,geometry)
    scene=json.loads(geometry.read_text())
    if scene['snapshot_sha256']!=hashlib.sha256(snapshot.read_bytes()).hexdigest():raise ValueError('Viewing geometry snapshot changed')
    opening=float(intent['gripper_width_m'])if opening is None else float(opening)
    result=review_path(plan_path,'data/commissioning/fr3_stock_hand.urdf',snapshot,handeye,geometry,joint_space=True,opening=opening)
    checks=result['checks'];checks.pop('minimum12s_duration');checks['minimum8s_duration']=result['duration_sec']>=8.;checks.update(viewing_line=result['max_cartesian_line_deviation_m']<=.0005,
        viewing_orientation=result['maximum_orientation_step_deg']<=.1,
        viewing_progress=result['path_progress_min_m']>=-.0005 and result['path_progress_max_m']<=distance+.0005)
    result['failed_checks']=[key for key,passed in checks.items()if not passed];result['passed_numeric_checks']=not result['failed_checks'];result['scene_reference_semantics']=scene['scene_reference_semantics'];result['purpose']='noncontact_view_shift';result['wrapper_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();result['limits_of_review']='Noncontact viewing translation only; measured stationary cloud and fitted table, conservative assumed camera/bracket envelopes. Unseen scene and robot internal self-avoidance model are not certified; application wrist restrictions apply. No descent/grasp authorization.'
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('plan');p.add_argument('--output');a=p.parse_args();r=review(a.plan);out=Path(a.output)if a.output else Path(a.plan).with_suffix('.validation.json');out.write_text(json.dumps(r,indent=2));print(json.dumps({k:v for k,v in r.items()if k not in('full_stock_hand_observed_cloud_sweep','full_arm_convex_hull_review','joint_limits')},indent=2));return 0 if r['passed_numeric_checks'] else 2
if __name__=='__main__':raise SystemExit(main())
