"""Offline source-footprint clearance after the explicitly reviewed upward carry."""
import argparse,hashlib,json
from pathlib import Path
import cv2,numpy as np,yaml
from lego_sorter.isolated_brick import load_frame,_cloud


def verify(snapshot,carry_result):
    carry=json.loads(Path(carry_result).read_text());plan=json.loads(Path(carry['plan_file']).read_text());meta=plan['isolated_diagnostic_stage']
    if meta.get('stage')!='carry_observe300'or carry.get('success')is not True or carry.get('isolated_lift10_result',{}).get('holding_width_verified')is not True:raise ValueError('Successful reviewed upward carry with measured holding width is required')
    hold=json.loads(Path(meta['hold_result']).read_text());grasp=json.loads(Path(hold['pose_plan']).read_text());gm=grasp['isolated_diagnostic_stage'];source=json.loads(Path(meta['targets']).read_text());plane=next(v for v in source['planes']if v['plane']==meta['plane'])
    frame,tcp=load_frame(snapshot);x=np.array(yaml.safe_load(Path(meta['handeye']).read_text())['T_tcp_color']);bc=tcp@x;normal=np.array(plane['measured_table_plane']['normal_base']);floor=np.array(plane['floor_ray_point_base_m']);grasp_pose=np.array(gm['physical_goal'])
    center=np.array(plane['assumed_nominal_stud_top_base_m'])-normal*.0114
    corners=np.array([center+grasp_pose[:3,0]*a+grasp_pose[:3,1]*b for a,b in [(-.01605,-.0079),(.01605,-.0079),(.01605,.0079),(-.01605,.0079)]])
    corners-=((corners-floor)@normal)[:,None]*normal
    camera=(corners-bc[:3,3])@bc[:3,:3];k=frame.intrinsics
    if any(abs(v)>1e-8 for v in k.get('coeffs',[])):raise ValueError('Source verifier requires the recorded zero-distortion aligned color model')
    pixels=np.c_[k['fx']*camera[:,0]/camera[:,2]+k['ppx'],k['fy']*camera[:,1]/camera[:,2]+k['ppy']]
    if camera[:,2].min()<.32 or np.any(pixels[:,0]<10)or np.any(pixels[:,1]<10)or np.any(pixels[:,0]>=frame.depth_m.shape[1]-10)or np.any(pixels[:,1]>=frame.depth_m.shape[0]-10):raise ValueError('Original source footprint is clipped or below320mm optical observation distance')
    # Shrink the original footprint so surrounding white table cannot hide a brick.
    inner=pixels.mean(axis=0)+(pixels-pixels.mean(axis=0))*.80
    mask=np.zeros(frame.depth_m.shape,'uint8');cv2.fillConvexPoly(mask,np.round(inner).astype(int),255)
    cloud,uv=_cloud(frame,1);inside=mask[uv[:,1],uv[:,0]]>0;points=cloud[inside]@bc[:3,:3].T+bc[:3,3];height=(points-floor)@normal
    total=int(np.count_nonzero(mask));valid=float(len(points)/total)if total else 0.
    if len(points)<100:raise ValueError('Insufficient current depth on original source footprint')
    table_fraction=float(np.mean(np.abs(height)<=.003));median=float(np.median(np.abs(height)));hsv=cv2.cvtColor(frame.color_bgr,cv2.COLOR_BGR2HSV);saturated=float(np.mean(hsv[:,:,1][mask>0]>80))
    checks={'valid_depth_fraction':valid>=.75,'measured_floor_fraction':table_fraction>=.90,'median_floor_residual':median<=.0025,'colored_brick_absent':saturated<=.05}
    return {'source_footprint_cleared':all(checks.values()),'held_brick_visually_observed':False,'holding_width_verified':True,'combined_pick_evidence':all(checks.values()),'checks':checks,'original_source_floor_corners_base_m':corners.tolist(),'projected_footprint_px':pixels.tolist(),'valid_depth_fraction':valid,'measured_floor_fraction':table_fraction,'median_absolute_floor_height_m':median,'high_saturation_fraction':saturated,'minimum_optical_depth_m':float(camera[:,2].min()),'snapshot':str(Path(snapshot).resolve()),'snapshot_sha256':hashlib.sha256(Path(snapshot).read_bytes()).hexdigest(),'carry_result':str(Path(carry_result).resolve()),'carry_result_sha256':hashlib.sha256(Path(carry_result).read_bytes()).hexdigest(),'limitations':'Combines original footprint now matching white measuredfloor with separately verified Hand width/upward motion. Does not claim direct visibility of the held brick, release, destination placement or production counts.'}


def main():
    p=argparse.ArgumentParser();p.add_argument('snapshot');p.add_argument('--carry-result',required=True);p.add_argument('--output',required=True);a=p.parse_args();r=verify(a.snapshot,a.carry_result);Path(a.output).write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2));return 0 if r['combined_pick_evidence'] else 2
if __name__=='__main__':raise SystemExit(main())
