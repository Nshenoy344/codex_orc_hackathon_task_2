"""One10mm isolated-brick test lift after verified Hand grasp; no finger action."""
import argparse,hashlib,json
from pathlib import Path
from itertools import product
import numpy as np,yaml
from scipy.spatial.transform import Rotation
from lego_sorter.geometry import matrix
from lego_sorter.isolated_brick import load_frame,_cloud
from validate_isolated_hover import review as review_path


def contract(plan_path):
    path=Path(plan_path);plan=json.loads(path.read_text());m=plan['isolated_diagnostic_stage']
    if m.get('stage')not in('lift10','carry_observe300','carry_template_view'):raise ValueError('Only10mm lift or upward carry_observe300 is enabled')
    h=json.loads(Path(m['hold_result']).read_text())
    if h.get('success')is not True or h.get('status')!='grasped_pending_lift'or h.get('hand_grasp_and_body_width_verified')is not True or h.get('brick_lifted_verified')is not False:raise ValueError('Successful matching pending-lift Hand proof is required')
    for key,hash_key in [('pose_plan','pose_plan_sha256'),('pose_result','pose_result_sha256')]:
        if hashlib.sha256(Path(h[key]).read_bytes()).hexdigest()!=h[hash_key]:raise ValueError('Holding pose evidence changed')
    grasp=json.loads(Path(h['pose_plan']).read_text());gm=grasp['isolated_diagnostic_stage']
    for key in('targets','snapshot','handeye'):
        if Path(m[key]).resolve()!=Path(gm[key]).resolve():raise ValueError('Lift evidence differs from actual grasp evidence')
    if m['plane']!=gm['plane']:raise ValueError('Lift floor differs from actual grasp floor')
    width=float(m['gripper_width_m'])
    if abs(width-float(h['gripper_after']['width_m']))>.0002 or abs(width-.0158)>.001:raise ValueError('Fresh lift width differs from verified holding body width')
    start=matrix(plan['T_base_tcp_start']);target=matrix(plan['T_base_tcp_target']);expected=start.copy()
    before=matrix(h['T_base_tcp_before'])
    if m['stage']=='carry_template_view':
        cr=json.loads(Path(m['carry_result']).read_text());cp=json.loads(Path(cr['plan_file']).read_text());proof=json.loads(Path(m['source_proof']).read_text())
        if cr.get('success')is not True or cr.get('isolated_lift10_result',{}).get('holding_width_verified')is not True or cp['isolated_diagnostic_stage']['stage']!='carry_observe300' or Path(cp['isolated_diagnostic_stage']['hold_result']).resolve()!=Path(m['hold_result']).resolve():raise ValueError('Matching successful source-observation carry required')
        if proof.get('combined_pick_evidence')is not True or Path(proof['carry_result']).resolve()!=Path(m['carry_result']).resolve()or proof['carry_result_sha256']!=hashlib.sha256(Path(m['carry_result']).read_bytes()).hexdigest():raise ValueError('Matching source-pick proof required')
        if Path(proof['snapshot']).resolve()!=Path(m['scene_snapshot']).resolve()or proof['snapshot_sha256']!=hashlib.sha256(Path(m['scene_snapshot']).read_bytes()).hexdigest():raise ValueError('Current scene differs from source-pick proof')
        hint=json.loads(Path(m['view_hint']).read_text());expected=matrix(hint['poses']['template_reference_view_hint']);before=matrix(cr['T_base_tcp_after'])
        if np.linalg.norm(expected[:3,3]-start[:3,3])>.400 or expected[2,3]<.300:raise ValueError('Template viewing hint exceeds400mm translation or is too low')
    elif m['stage']=='carry_observe300':
        lr=json.loads(Path(m['lift10_result']).read_text());lp=json.loads(Path(lr['plan_file']).read_text())
        if lr.get('success')is not True or lr.get('isolated_lift10_result',{}).get('success')is not True or lp.get('isolated_diagnostic_stage',{}).get('stage')!='lift10' or Path(lp['isolated_diagnostic_stage']['hold_result']).resolve()!=Path(m['hold_result']).resolve():raise ValueError('Successful matching10mm lift required before carry')
        before=matrix(lr['T_base_tcp_after']);expected[2,3]=.300
        if not 0<expected[2,3]-start[2,3]<=.350:raise ValueError('Carry must rise atmost350mm toTCPZ300mm')
    else:expected[2,3]+=.010
    # Goal construction and typed planning start use separate fresh TF samples.
    # Bound their observed micrometre jitter without permitting a lateral move.
    z_tolerance=1e-8 if m['stage']in('carry_observe300','carry_template_view')else .00005
    if (np.linalg.norm(target[:2,3]-expected[:2,3])>.00005 or
            abs(target[2,3]-expected[2,3])>z_tolerance or
            Rotation.from_matrix(expected[:3,:3].T@target[:3,:3]).magnitude()>.0001):
        raise ValueError('Lift target must preserveXY/R within50um/100urad sampling tolerance and declared upwardZ')
    if m['stage']in('carry_observe300','carry_template_view')and 'holding_checkpoint'in m:
        checkpoint=m['holding_checkpoint'];fresh=matrix(checkpoint['T_base_tcp'])
        checkpoint_key='carry_result'if m['stage']=='carry_template_view'else'lift10_result'
        if checkpoint.get(checkpoint_key+'_sha256')!=hashlib.sha256(Path(m[checkpoint_key]).read_bytes()).hexdigest():raise ValueError('Fresh holding checkpoint source changed')
        if np.linalg.norm(fresh[:3,3]-before[:3,3])>.003 or Rotation.from_matrix(fresh[:3,:3].T@before[:3,:3]).magnitude()>.01:raise ValueError('Holding deflection exceeds3mm/.01rad checkpoint envelope')
        if abs(float(checkpoint['width_m'])-width)>.0002:raise ValueError('Fresh checkpoint opening differs from lift plan')
        before=fresh
    if np.linalg.norm(start[:3,3]-before[:3,3])>.001 or Rotation.from_matrix(start[:3,:3].T@before[:3,:3]).magnitude()>.01:raise ValueError('Arm moved since verified holding checkpoint')
    source=json.loads(Path(m['targets']).read_text());plane=next(p for p in source['planes']if p['plane']==m['plane'])
    for key,source_key,hash_key in [('snapshot','snapshot','snapshot_sha256'),('handeye','handeye_artifact','handeye_sha256')]:
        if Path(m[key]).resolve()!=Path(source[source_key]).resolve()or hashlib.sha256(Path(m[key]).read_bytes()).hexdigest()!=source[hash_key]:raise ValueError('Nominal source evidence changed')
    roi=np.array(m['source_roi_px'],int)
    if roi.shape!=(4,)or not np.array_equal(roi,[690,480,110,160]):raise ValueError('Only the inspected isolated-object sourceROI can be excluded')
    return plan,m,h,grasp,plane


def review(plan_path):
    path=Path(plan_path);plan,m,holding,grasp,plane=contract(path);measured=plane['measured_table_plane'];normal=np.array(measured['normal_base']);normal/=np.linalg.norm(normal);floor=np.array(plane['floor_ray_point_base_m'])
    geom=path.with_suffix('.scene_geometry.json');g={'visible_top_base_m':plane['assumed_nominal_stud_top_base_m'],'table_point_base_m':floor.tolist(),'measured_table_plane':{'normal_base':normal.tolist(),**measured['quality']},'source_targets_sha256':hashlib.sha256(Path(m['targets']).read_bytes()).hexdigest(),'semantics':'Measuredfloor and nominalheld2x4 boundingbox; designed initial tablecontact is not free clearance.'}
    if geom.exists()and json.loads(geom.read_text())!=g:raise ValueError('Lift review geometry changed')
    geom.write_text(json.dumps(g,indent=2))
    scene_snapshot=m['scene_snapshot']if m['stage']=='carry_template_view'else m['snapshot']
    boxes=None
    if m['stage']=='carry_template_view':
        from lego_sorter.environment_boxes import load_boxes
        boxes=load_boxes(m['environment_boxes'])
    r=review_path(path,'data/commissioning/fr3_stock_hand.urdf',scene_snapshot,m['handeye'],geom,joint_space=True,opening=float(m['gripper_width_m']),camera_mode='oriented_contact',include_dense_poses=True,environment_boxes=boxes)
    if m['stage']=='carry_template_view':
        r['scene_snapshot']=r['snapshot'];r['scene_snapshot_sha256']=r['snapshot_sha256'];r['snapshot']=str(Path(m['snapshot']).resolve());r['snapshot_sha256']=hashlib.sha256(Path(m['snapshot']).read_bytes()).hexdigest()

    poses=np.asarray(r.pop('_dense_tcp_poses'));grasp_pose=matrix(grasp['isolated_diagnostic_stage']['physical_goal'])
    # Overall nominal11.4mm envelope includes studs. This is not a measured attached mesh.
    center=np.array(plane['assumed_nominal_stud_top_base_m'])-normal*.0057
    center_local=(center-grasp_pose[:3,3])@grasp_pose[:3,:3]
    corners=center_local+np.array(list(product((-.01605,.01605),(-.0079,.0079),(-.0057,.0057))))
    frame,tcp=load_frame(scene_snapshot);x=np.array(yaml.safe_load(Path(m['handeye']).read_text())['T_tcp_color']);bc=tcp@x;cloud,pixels=_cloud(frame,3);world=cloud@bc[:3,:3].T+bc[:3,3]
    rx,ry,rw,rh=m['source_roi_px'];outside=~((pixels[:,0]>=rx)&(pixels[:,0]<rx+rw)&(pixels[:,1]>=ry)&(pixels[:,1]<ry+rh));above=(world-floor)@normal>.002;outside=np.ones(len(world),bool)if m['stage']=='carry_template_view'else outside;obstacles=world[outside&above]
    heights=[];counts=[];held_environment_hits=[]
    for pose in poses:
        corners_base=corners@pose[:3,:3].T+pose[:3,3];heights.append(float(((corners_base-floor)@normal).min()))
        if boxes is not None:
            from lego_sorter.environment_boxes import intersections
            held_environment_hits.extend(intersections(corners_base,boxes))
        local=(obstacles-pose[:3,3])@pose[:3,:3]-center_local
        counts.append(int(np.count_nonzero(np.all(np.abs(local)<=np.array([.01605,.0079,.0057])+.001,axis=1))))
    heights=np.asarray(heights);checks=r['checks'];checks.pop('minimum150mm_tip_clearance');checks.update(lift_straight_line=r['max_cartesian_line_deviation_m']<=.0005,lift_progress=r['path_progress_min_m']>=-.0005 and r['path_progress_max_m']<=r['path_length_m']+.0005,
        held_box_no_new_obstacle=max(counts)==0,held_box_initial_nonpenetration_with_uncertainty=float(heights.min())>=-.002,
        held_box_monotonic_floor_clearance=float(np.diff(heights).min())>=-.00001,held_box_final_floor_gain=float(heights[-1]-heights[0])>=float(plan['T_base_tcp_target'][2][3]-plan['T_base_tcp_start'][2][3])-.001)
    if m['stage']=='carry_observe300':
        checks.pop('held_box_initial_nonpenetration_with_uncertainty')
        checks['held_box_carry_positive_floor_clearance']=float(heights.min())>=.003
        checks['minimum15s_duration']=r['duration_sec']>=15.

    if m['stage']=='carry_template_view':
        for key in('held_box_initial_nonpenetration_with_uncertainty','held_box_monotonic_floor_clearance','held_box_final_floor_gain'):checks.pop(key)
        checks.update(held_box_view_floor_clearance=float(heights.min())>=.200,template_view_tcp_height=r['minimum_tcp_z_m']>=.250,
            minimum15s_duration=r['duration_sec']>=15.,held_box_recorded_environment_clear=not held_environment_hits)
        r['held_environment_box_collisions']=held_environment_hits
        r['environment_helper_sha256']=hashlib.sha256((Path(__file__).resolve().parents[1]/'lego_sorter/environment_boxes.py').read_bytes()).hexdigest()
    r['held_nominal_brick']={'dimensions_m':[.0321,.0158,.0114],'center_tcp_m':center_local.tolist(),'initial_floor_clearance_m':float(heights[0]),'minimum_floor_clearance_m':float(heights.min()),'final_floor_clearance_m':float(heights[-1]),'maximum_obstacle_intersections':max(counts),'floor_model_uncertainty_allowance_m':.002,'uncertainty_semantics':'Explicit2mm allowance for localfloor fit, pose residual, nominal brick dimensions and initial support contact; not measured positive clearance. Floorpoints≤2mm and originalbrickROI are excluded only from held-object self/initial-support collision test; fullarm/Hand/camera checks still use originalcloud.'}
    r['checks']={k:bool(v)for k,v in checks.items()};r['failed_checks']=[k for k,v in r['checks'].items()if not v];r['passed_numeric_checks']=not r['failed_checks'];r['purpose']='isolated_'+m['stage'];r['wrapper_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    extra=('view_hint','carry_result','source_proof','environment_boxes')if m['stage']=='carry_template_view'else(('lift10_result',)if m['stage']=='carry_observe300'else())
    for key in('targets','hold_result')+extra:
        r[key]=str(Path(m[key]).resolve());r[key+'_sha256']=hashlib.sha256(Path(m[key]).read_bytes()).hexdigest()
    r['limits_of_review']=('Only the pinned raised template-view hint while preserving the verified Hand hold; lateral translation and orientation interpolation are explicitly reviewed against current scene and recorded input obstacles. This is a viewing pose, not a measured destination; no finger action, release or production counts.' if m['stage']=='carry_template_view' else 'Only the declared upward lift/carry with existing verified Hand hold. No finger action, lateral/yaw move, release or production counts; actual visual pick proof remains separate.')
    return r


def main():
    p=argparse.ArgumentParser();p.add_argument('plan');a=p.parse_args();r=review(a.plan);Path(a.plan).with_suffix('.validation.json').write_text(json.dumps(r,indent=2));print(json.dumps({k:v for k,v in r.items()if k not in('joint_limits','full_stock_hand_observed_cloud_sweep','full_arm_convex_hull_review')},indent=2));return 0 if r['passed_numeric_checks'] else 2
if __name__=='__main__':raise SystemExit(main())
