"""Independent static-released-brick/Hand review of one vertical retreat only."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np,yaml
from scipy.spatial.transform import Rotation
from lego_sorter.geometry import matrix
from lego_sorter.released_brick_geometry import static_bounds,review_hand
from lego_sorter.environment_boxes import load_boxes
from validate_isolated_hover import review as review_path


def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def contract(path):
 p=json.loads(Path(path).read_text());m=p['isolated_diagnostic_stage'];r=json.loads(Path(m['release_result']).read_text());t=json.loads(Path(m['placement_target']).read_text())
 if m.get('stage')!='postrelease_retreat350'or m.get('gripper_action_authorized')is not False or m.get('attached_model_removed')is not True or p.get('planning_scene_readback_verified')is not True:raise ValueError('Only verified static-model vertical postrelease retreat enabled')
 if r.get('success')is not True or r.get('opening_width_verified')is not True or r.get('status')!='released_pending_visual_verification'or abs(float(m['gripper_width_m'])-.020)>.0002 or abs(float(m['gripper_width_m'])-float(r['gripper_after']['width_m']))>.0002:raise ValueError('Verified20mm release required')
 for key in('pose_plan','pose_result','placement_target','pose_review'):
  if digest(r[key])!=r[key+'_sha256']:raise ValueError('Original release evidence changed')
 if Path(r['placement_target']).resolve()!=Path(m['placement_target']).resolve():raise ValueError('Retreat target differs from released target')
 start=matrix(p['T_base_tcp_start']);target=matrix(p['T_base_tcp_target']);anchor,lo,hi=static_bounds(r,t)
 if np.linalg.norm(start[:3,3]-anchor[:3,3])>.003 or Rotation.from_matrix(start[:3,:3].T@anchor[:3,:3]).magnitude()>.01:raise ValueError('Retreat start differs from actual release pose')
 if np.linalg.norm(target[:2,3]-start[:2,3])>.00005 or abs(target[2,3]-.350)>1e-8 or Rotation.from_matrix(start[:3,:3].T@target[:3,:3]).magnitude()>.0001 or not 0<target[2,3]-start[2,3]<=.350:raise ValueError('Retreat must preserveXY/R and rise to350mm')
 return p,m,r,t,anchor,lo,hi


def review(path):
 path=Path(path);p,m,released,target,anchor,lo,hi=contract(path);prior=json.loads(Path(released['pose_review']).read_text());g=json.loads(Path(prior['geometry_report']).read_text());geometry=path.with_suffix('.scene_geometry.json');geometry.write_text(json.dumps(g,indent=2));boxes=load_boxes(m['environment_boxes'])
 r=review_path(path,'data/commissioning/fr3_stock_hand.urdf',m['scene_snapshot'],m['handeye'],geometry,joint_space=True,opening=float(m['gripper_width_m']),camera_mode='oriented_contact',include_dense_poses=True,environment_boxes=boxes);poses=np.array(r.pop('_dense_tcp_poses'));static=review_hand(poses,float(m['gripper_width_m']),anchor,lo,hi,camera_extrinsic=np.array(yaml.safe_load(Path(m['handeye']).read_text())['T_tcp_color']))
 checks=r['checks'];checks.pop('minimum150mm_tip_clearance');checks.update(minimum15s_duration=r['duration_sec']>=15.,released_static_brick_hand_clear=static['static_brick_hand_clear'],retreat_vertical_line=r['max_cartesian_line_deviation_m']<=.0005,retreat_bounded_progress=r['path_progress_min_m']>=-.0005 and r['path_progress_max_m']<=r['path_length_m']+.0005,retreat_monotonic_up=float(np.diff(poses[:,2,3]).min())>=-.00001)
 r['release_anchor_acquisition_distance_m']=float(np.linalg.norm(np.array(p['T_base_tcp_start'])[:3,3]-anchor[:3,3]))
 r.update(checks={k:bool(v)for k,v in checks.items()},purpose='isolated_postrelease_retreat350',released_static_brick_review=static,wrapper_sha256=digest(__file__),released_geometry_module_sha256=digest(Path(__file__).resolve().parents[1]/'lego_sorter/released_brick_geometry.py'),environment_boxes_module_sha256=digest(Path(__file__).resolve().parents[1]/'lego_sorter/environment_boxes.py'),limits_of_review='Vertical retreat only after native20mm opening. Released brick is conditional STATIC, not attached;0..3mm settling and1mm proxy margin. No opening/release/counts or placement verification.')
 for key in('release_result','placement_target','scene_snapshot','handeye','environment_boxes'):r[key]=m[key];r[key+'_sha256']=digest(m[key])
 r['failed_checks']=[k for k,v in r['checks'].items()if not v];r['passed_numeric_checks']=not r['failed_checks'];return r

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('plan');a=p.parse_args();r=review(a.plan);Path(a.plan).with_suffix('.validation.json').write_text(json.dumps(r,indent=2));print(json.dumps({k:v for k,v in r.items()if k not in('joint_limits','full_stock_hand_observed_cloud_sweep','full_arm_convex_hull_review')},indent=2));raise SystemExit(0 if r['passed_numeric_checks']else 2)
