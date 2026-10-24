"""Offline isolated recovery geometry from actual post-placement RGBD only."""
import json,hashlib,argparse
from pathlib import Path
import numpy as np,cv2,yaml
from lego_sorter.camera import Frame
from lego_sorter.pallet import fit_surface_plane
parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--snapshot',default='data/commissioning/first_placement_after_retract.npz');parser.add_argument('--output',default='data/commissioning/misplaced_maroon_recovery_target.json');parser.add_argument('--polygon',default='[[698,480],[740,481],[740,565],[696,564]]',help='JSON actualvisibleface polygon');args=parser.parse_args()
P=Path('data/commissioning');snapshot=Path(args.snapshot);z=np.load(snapshot);k=json.loads(str(z['intrinsics_json']));f=Frame(z['color_bgr'],z['depth_m'],k,float(z['captured_monotonic']));hand=P/'charuco_rgbd_plane_candidate.yaml';B=np.array(z['T_base_tcp'])@np.array(yaml.safe_load(hand.read_text())['T_tcp_color'])
polygon=np.array(json.loads(args.polygon),dtype=int);mask=np.zeros(f.depth_m.shape,np.uint8);cv2.fillPoly(mask,[polygon],1);mask=cv2.erode(mask,np.ones((15,15),np.uint8));mask=mask.astype(bool)&np.isfinite(f.depth_m)&(f.depth_m>0)
p,n,rms=fit_surface_plane(f,mask);rows,cols=np.nonzero(mask);points=np.array([f.deproject(x,y,f.depth_m[y,x])for y,x in zip(rows,cols)]);nbase=B[:3,:3]@n
if nbase[2]<0:n=-n;nbase=-nbase
localnormal=nbase.copy()
prior=json.loads((P/'first_loose_placement_target.json').read_text());env=json.loads((P/'template_upper_envelope_review.json').read_text());B0=np.array(prior['T_base_color_reference']);n0=B0[:3,:3]@np.array(env['normal_camera']);n0/=np.linalg.norm(n0)
nbase=n0;n=B[:3,:3].T@nbase
h=np.percentile(points@n,[50,95,99,100]);p=p+n*(h[0]-n@p);centerpx=polygon.mean(0)
def rayplane(pixel):
 ray=f.deproject(*pixel,1.);return ray*((n@p)/(n@ray))
center=B[:3,:3]@rayplane(centerpx)+B[:3,3];end1=rayplane((polygon[0]+polygon[1])/2);end2=rayplane((polygon[2]+polygon[3])/2);a=B[:3,:3]@(end2-end1);a-=nbase*(a@nbase);a/=np.linalg.norm(a)
# Keep tool down against observed surface. Choose equivalentyaw closest current.
def rot(a):return np.column_stack([a,np.cross(-nbase,a),-nbase])
R=rot(a)
if np.trace(B[:3,:3].T@rot(-a))>np.trace(B[:3,:3].T@R):R=rot(-a)
upper=float(h[2]+nbase@B[:3,3]);toe=np.array([0,0,.0095]);tcp=center.copy();tcp+=nbase*(upper-.003-nbase@(tcp+R@toe));T=np.eye(4);T[:3,:3]=R;T[:3,3]=tcp;A=T.copy();A[2,3]+=.03;L=T.copy();L[2,3]+=.01
corners=np.array([[x,s*(.030/2+t),.0095]for x in(-.010,.010)for s in(-1,1)for t in(0,.0135)]);worldcorners=corners@R.T+tcp
prior=json.loads((P/'first_loose_placement_target.json').read_text());env=json.loads((P/'template_upper_envelope_review.json').read_text());B0=np.array(prior['T_base_color_reference']);n0=B0[:3,:3]@np.array(env['normal_camera']);table=B0[:3,:3]@np.array(env['table_point_camera_m'])+B0[:3,3];lower=max(s['height_percentiles_m']['max']for s in env['samples']);loweroffset=n0@table+lower
out=dict(kind='isolated_misplaced_brick_recovery_target_only',snapshot=str(snapshot.resolve()),visible_face_polygon_px=polygon.tolist(),surface_plane_rms_m=rms,surface_valid_pixels=len(points),normal_base=nbase.tolist(),local_face_fit_normal_base_diagnostic=localnormal.tolist(),normal_source='Independent broad template support plane; local smallface slope not used',visible_upper_p99_offset_base_m=upper,visible_surface_percentiles_camera_m=h.tolist(),observed_face_center_base_m=center.tolist(),T_base_tcp_grasp=T.tolist(),T_base_tcp_approach30=A.tolist(),T_base_tcp_recoverylift10=L.tolist(),gripper_opening_m=.030,body_grasp_prototype=True,body_stud_semantics_verified=False,nominal_height_subtracted=False,production_ready=False,action_sent=False,toe_center_insertion_below_observed_p99_m=.003,full_toe_insertion_range_below_p99_m=[float(min(upper-worldcorners@nbase)),float(max(upper-worldcorners@nbase))],minimum_toe_clearance_above_prior_lower_layer_globalmax_m=float(min(worldcorners@n0-loweroffset)),lower_layer_bound_is_prior_observation=True,required_independent_review='Actual fullHand/arm/camera/cloud/staticboxes and lower-neighbor geometry; no gripper action authorized by target',source_files=[{'path':str(path.resolve()),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}for path in[snapshot,hand,P/'first_loose_placement_target.json',P/'template_upper_envelope_review.json']])
stud=T.copy();stud[:3,3]+=nbase*.004
out['optional_stud_initial_above_upper_pose']=stud.tolist()
out['optional_stud_trial_protocol']={'isolated_prototype_only':True,'initial_toe_above_measured_p99_m':.001,'opening_m':.0168,'descent_step_m':.0002,'max_total_descent_m':.0018,'grasp_goal_m':.0128,'epsilon_outer_m':.0008,'never_descend_closed':True,'requires_independent_per_step_review':True}
out['depth_camera_metadata']=json.loads(str(z['metadata_json']))
out['observed_interior_valid_fraction']=float(len(points)/max(1,np.count_nonzero(cv2.erode(cv2.fillPoly(np.zeros(f.depth_m.shape,np.uint8),[polygon],1),np.ones((15,15),np.uint8)))))
oldpath=P/'misplaced_maroon_recovery_target.json'
if oldpath.exists() and Path(args.output).resolve()!=oldpath.resolve():
 old=json.loads(oldpath.read_text());out['comparison_to_original_target']={'path':str(oldpath.resolve()),'center_delta_m':(center-np.array(old['observed_face_center_base_m'])).tolist(),'tcp_grasp_delta_m':(tcp-np.array(old['T_base_tcp_grasp'])[:3,3]).tolist(),'upper_offset_delta_m':upper-old['visible_upper_p99_offset_base_m']}
if Path(args.output).exists():raise RuntimeError('Refusing to overwrite existing pinned target')
Path(args.output).write_text(json.dumps(out,indent=2));print(json.dumps({k:out[k]for k in['surface_plane_rms_m','T_base_tcp_grasp','full_toe_insertion_range_below_p99_m','minimum_toe_clearance_above_prior_lower_layer_globalmax_m']},indent=2))
