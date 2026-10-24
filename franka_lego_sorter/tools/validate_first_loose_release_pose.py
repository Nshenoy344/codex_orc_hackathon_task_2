"""Independent dense review of final loose-placement pose only; no opening."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np,yaml
from scipy.spatial.transform import Rotation
from lego_sorter.held_view_checkpoint import verify_checkpoint
from lego_sorter.isolated_brick import load_frame,_cloud
from lego_sorter.environment_boxes import load_boxes,intersections
from validate_isolated_hover import review as review_path

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def command_matches_expected(command, expected):
 command=np.asarray(command,float);expected=np.asarray(expected,float)
 return bool(np.linalg.norm(command[:3,3]-expected[:3,3])<=.00005 and Rotation.from_matrix(command[:3,:3].T@expected[:3,:3]).magnitude()<=.0001)

def review(path):
 path=Path(path);plan=json.loads(path.read_text());m=plan['isolated_diagnostic_stage']
 if m.get('stage')!='first_loose_release_pose' or m.get('gripper_action_authorized')is not False:raise ValueError('Only no-opening first_loose_release_pose accepted')
 for key in ('targets','snapshot','handeye','reference_plan','reference_result'):
  if digest(m[key])!=m[key+'_sha256']:raise ValueError('Plan source changed: '+key)
 if not np.allclose(m['physical_goal'],json.loads(Path(m['targets']).read_text())['T_base_tcp_release'],atol=1e-9):raise ValueError('Physical goal changed')
 target=json.loads(Path(m['targets']).read_text())
 for item in target['source_files']:
  if digest(item['path'])!=item['sha256']:raise ValueError('Placement evidence changed')
 expected_scene=next(x['path']for x in target['source_files']if Path(x['path']).name=='template_reference_uncovered_paired.npz')
 expected_handeye=next(x['path']for x in target['source_files']if Path(x['path']).name=='charuco_rgbd_plane_candidate.yaml')
 if Path(m['snapshot']).resolve()!=Path(expected_scene).resolve()or Path(m['handeye']).resolve()!=Path(expected_handeye).resolve():raise ValueError('Scene/calibration must match pinned placement evidence')
 physical=np.array(target['T_base_tcp_release']);start=np.array(plan['T_base_tcp_start']);command=np.array(plan['T_base_tcp_target'])
 reference=json.loads(Path(m['reference_plan']).read_text());result=json.loads(Path(m['reference_result']).read_text())
 if result.get('success')is not True or Path(result['plan_file']).resolve()!=Path(m['reference_plan']).resolve():raise ValueError('Successful approach execution required')
 if reference['isolated_diagnostic_stage']['stage']!='first_loose_place_approach' or not np.allclose(reference['T_base_tcp_target'],target['T_base_tcp_approach30'],atol=1e-9):raise ValueError('Reference must be this exact approach30')
 observed=np.array(result['T_base_tcp_after'])
 if np.linalg.norm(start[:3,3]-observed[:3,3])>.001 or Rotation.from_matrix(start[:3,:3].T@observed[:3,:3]).magnitude()>.01:raise ValueError('Robot moved since approach')
 goal=np.array(reference['T_base_tcp_target']);bias=goal[:3,3]-start[:3,3];turn=goal[:3,:3]@start[:3,:3].T
 if np.linalg.norm(bias)>.003 or Rotation.from_matrix(turn).magnitude()>.01:raise ValueError('Approach tracking bias exceeds3mm/.01rad')
 expected=physical.copy();expected[:3,3]+=bias;expected[:3,:3]=turn@physical[:3,:3]
 if not command_matches_expected(command,expected):raise ValueError('Command differs from measured bias compensation beyond50um/100urad sampling jitter')
 if not 0<start[2,3]-command[2,3]<=.035 or np.linalg.norm(start[:3,3]-command[:3,3])>.035:raise ValueError('Only descending segment<=35mm allowed')
 if not np.isclose(target['minimum_clearance_above_p99_m'],.003,atol=1e-8):raise ValueError('Physical goal requires exact3mm p99 gap')
 if abs(float(m['gripper_width_m'])-.01535)>.0003:raise ValueError('Measured held width required')
 m['scene_snapshot']=m['snapshot'];m['placement_target']=m['targets']
 apmeta=reference['isolated_diagnostic_stage'];m['environment_boxes']=apmeta['environment_boxes'];m['environment_boxes_sha256']=apmeta['environment_boxes_sha256']
 if digest(m['environment_boxes'])!=m['environment_boxes_sha256']:raise ValueError('Environment changed')
 envelope_path=next(s['path'] for s in target['source_files']if Path(s['path']).name=='template_upper_envelope_review.json');envelope=json.loads(Path(envelope_path).read_text());B=np.array(target['T_base_color_reference']);n=B[:3,:3]@np.array(envelope['normal_camera']);table=B[:3,:3]@np.array(envelope['table_point_camera_m'])+B[:3,3]
 if n[2]<0:n=-n
 upper_offset=float(n@table+next(s['height_percentiles_m']['max']for s in envelope['samples']if s['slot_id']==target['slot_id']))
 g=dict(visible_top_base_m=(table+n*(upper_offset-n@table)).tolist(),table_point_base_m=table.tolist(),measured_table_plane=dict(normal_base=n.tolist()),semantics='Measured template surroundingtable and global actual upper-envelope, not bodytop')
 geometry=path.with_suffix('.scene_geometry.json');geometry.write_text(json.dumps(g,indent=2));boxes=load_boxes(m['environment_boxes'])
 r=review_path(path,'data/commissioning/fr3_stock_hand.urdf',m['scene_snapshot'],m['handeye'],geometry,joint_space=True,opening=float(m['gripper_width_m']),camera_mode='oriented_contact',include_dense_poses=True,environment_boxes=boxes)
 poses=np.array(r.pop('_dense_tcp_poses'));bottom=np.array(target['estimated_bottom_footprint_tcp_m']);bottomnormal=np.array(target['bottom_plane_tcp']['normal']);corners=np.vstack((bottom,bottom+bottomnormal*.0114))
 frame,tcp=load_frame(m['scene_snapshot']);X=np.array(yaml.safe_load(Path(m['handeye']).read_text())['T_tcp_color']);cloud,_=_cloud(frame,3);world=cloud@(tcp@X)[:3,:3].T+(tcp@X)[:3,3];center=corners.mean(axis=0);low=corners.min(axis=0)-.001;high=corners.max(axis=0)+.001
 clearances=[];counts=[];hits=[]
 for pose in poses:
  vertices=corners@pose[:3,:3].T+pose[:3,3];clearances.append(float((vertices@n-upper_offset).min()));hits.extend(intersections(vertices,boxes));local=(world-pose[:3,3])@pose[:3,:3];counts.append(int(np.all((local>=low)&(local<=high),axis=1).sum()))
 checks=r['checks'];checks.pop('minimum150mm_tip_clearance');checks.update(minimum12s_duration=r['duration_sec']>=12.,held_box1mm_above_slot_observed_max=min(clearances)>=.001,held_box_observed_cloud_clear=max(counts)==0,held_box_static_input_boxes_clear=not hits,approach_cartesian_line=r['max_cartesian_line_deviation_m']<=.0005,approach_bounded_progress=r['path_progress_min_m']>=-.0005 and r['path_progress_max_m']<=r['path_length_m']+.0005)
 r.update(purpose='isolated_first_loose_release_pose',checks={k:bool(v)for k,v in checks.items()},minimum_held_box_template_clearance_m=min(clearances),held_box_maximum_cloud_intersections=max(counts),held_environment_box_collisions=hits,held_box_nominal_height_m=.0114,global_observed_template_upper_offset_m=upper_offset,wrapper_sha256=digest(__file__),limits_of_review='Exact bounded final descent only, verified heldbody and conditional no-slip heldbox. Candidate calibration remains unverified. No finger action, release, counts or contact authorization.')
 for key in ('targets','snapshot','handeye','reference_plan','reference_result'):r[key]=m[key];r[key+'_sha256']=m[key+'_sha256']
 r['environment_boxes_module_sha256']=digest('lego_sorter/environment_boxes.py');r['environment_boxes']=m['environment_boxes'];r['environment_boxes_sha256']=m['environment_boxes_sha256'];r['physical_goal']=physical.tolist();r['strict_physical_position_tolerance_m']=.001;r['strict_physical_rotation_tolerance_rad']=.01
 r['failed_checks']=[k for k,v in r['checks'].items()if not v];r['passed_numeric_checks']=not r['failed_checks'];return r
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('plan');a=p.parse_args();r=review(a.plan);Path(a.plan).with_suffix('.validation.json').write_text(json.dumps(r,indent=2));print(json.dumps({k:v for k,v in r.items()if k not in ('joint_limits','full_stock_hand_observed_cloud_sweep','full_arm_convex_hull_review')},indent=2));raise SystemExit(0 if r['passed_numeric_checks']else 2)
