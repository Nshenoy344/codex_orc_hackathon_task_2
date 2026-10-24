"""Independent dense review of exact first loose-placement approach only."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np,yaml
from lego_sorter.held_view_checkpoint import verify_checkpoint
from lego_sorter.isolated_brick import load_frame,_cloud
from lego_sorter.environment_boxes import load_boxes,intersections
from validate_isolated_hover import review as review_path

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def review(path):
 path=Path(path);plan=json.loads(path.read_text());m=plan['isolated_diagnostic_stage']
 if m.get('stage')!='first_loose_place_approach' or m.get('gripper_action_authorized')is not False:raise ValueError('Only no-release first_loose_place_approach accepted')
 for key in ('checkpoint','placement_target','scene_snapshot','handeye','environment_boxes'):
  if digest(m[key])!=m[key+'_sha256']:raise ValueError('Plan source changed: '+key)
 c=verify_checkpoint(m['checkpoint'],np.array(plan['T_base_tcp_start']),float(m['gripper_width_m']));target=json.loads(Path(m['placement_target']).read_text())
 if target.get('kind')!='first_loose_placement_target_only'or target.get('production_placement_ready')is not False:raise ValueError('Unqualified exact placement target required')
 for item in target['source_files']:
  if digest(item['path'])!=item['sha256']:raise ValueError('Placement evidence changed')
 if not np.allclose(plan['T_base_tcp_target'],target['T_base_tcp_approach30'],atol=1e-9):raise ValueError('Target differs from exact approach30')
 if Path(m['handeye']).resolve()!=Path(c['handeye']).resolve()or Path(m['environment_boxes']).resolve()!=Path(c['environment_boxes']).resolve():raise ValueError('Hold/calibration/environment associations differ')
 expected_scene=next(s['path'] for s in target['source_files']if Path(s['path']).name=='template_reference_uncovered_paired.npz')
 if Path(m['scene_snapshot']).resolve()!=Path(expected_scene).resolve():raise ValueError('Scene differs from actual registered reference capture')
 envelope_path=next(s['path'] for s in target['source_files']if Path(s['path']).name=='template_upper_envelope_review.json');envelope=json.loads(Path(envelope_path).read_text());B=np.array(target['T_base_color_reference']);n=B[:3,:3]@np.array(envelope['normal_camera']);table=B[:3,:3]@np.array(envelope['table_point_camera_m'])+B[:3,3]
 if n[2]<0:n=-n
 upper_offset=float(n@table+max(s['height_percentiles_m']['max']for s in envelope['samples']))
 g=dict(visible_top_base_m=(table+n*(upper_offset-n@table)).tolist(),table_point_base_m=table.tolist(),measured_table_plane=dict(normal_base=n.tolist()),semantics='Measured template surroundingtable and global actual upper-envelope, not bodytop')
 geometry=path.with_suffix('.scene_geometry.json');geometry.write_text(json.dumps(g,indent=2));boxes=load_boxes(m['environment_boxes'])
 r=review_path(path,'data/commissioning/fr3_stock_hand.urdf',m['scene_snapshot'],m['handeye'],geometry,joint_space=True,opening=float(m['gripper_width_m']),camera_mode='oriented_contact',include_dense_poses=True,environment_boxes=boxes)
 poses=np.array(r.pop('_dense_tcp_poses'));bottom=np.array(target['estimated_bottom_footprint_tcp_m']);bottomnormal=np.array(target['bottom_plane_tcp']['normal']);corners=np.vstack((bottom,bottom+bottomnormal*.0114))
 frame,tcp=load_frame(m['scene_snapshot']);X=np.array(yaml.safe_load(Path(m['handeye']).read_text())['T_tcp_color']);cloud,_=_cloud(frame,3);world=cloud@(tcp@X)[:3,:3].T+(tcp@X)[:3,3];center=corners.mean(axis=0);low=corners.min(axis=0)-.001;high=corners.max(axis=0)+.001
 clearances=[];counts=[];hits=[]
 for pose in poses:
  vertices=corners@pose[:3,:3].T+pose[:3,3];clearances.append(float((vertices@n-upper_offset).min()));hits.extend(intersections(vertices,boxes));local=(world-pose[:3,3])@pose[:3,:3];counts.append(int(np.all((local>=low)&(local<=high),axis=1).sum()))
 checks=r['checks'];checks.pop('minimum150mm_tip_clearance');checks.update(minimum15s_duration=r['duration_sec']>=15.,held_box20mm_above_global_observed_template_max=min(clearances)>=.020,held_box_observed_cloud_clear=max(counts)==0,held_box_static_input_boxes_clear=not hits,approach_cartesian_line=r['max_cartesian_line_deviation_m']<=.0005,approach_bounded_progress=r['path_progress_min_m']>=-.0005 and r['path_progress_max_m']<=r['path_length_m']+.0005)
 r.update(purpose='isolated_first_loose_place_approach',checks={k:bool(v)for k,v in checks.items()},minimum_held_box_template_clearance_m=min(clearances),held_box_maximum_cloud_intersections=max(counts),held_environment_box_collisions=hits,held_box_nominal_height_m=.0114,global_observed_template_upper_offset_m=upper_offset,wrapper_sha256=digest(__file__),limits_of_review='Exact raised approach only, verified heldbody and conditional no-slip heldbox. Candidate calibration remains unverified. No finger action, release, counts or contact authorization.')
 r['environment_helper_sha256']=digest(Path(__file__).resolve().parents[1]/'lego_sorter/environment_boxes.py')
 for key in ('checkpoint','placement_target','scene_snapshot','handeye','environment_boxes'):r[key]=m[key];r[key+'_sha256']=m[key+'_sha256']
 r['failed_checks']=[k for k,v in r['checks'].items()if not v];r['passed_numeric_checks']=not r['failed_checks'];return r
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('plan');a=p.parse_args();r=review(a.plan);Path(a.plan).with_suffix('.validation.json').write_text(json.dumps(r,indent=2));print(json.dumps({k:v for k,v in r.items()if k not in ('joint_limits','full_stock_hand_observed_cloud_sweep','full_arm_convex_hull_review')},indent=2));raise SystemExit(0 if r['passed_numeric_checks']else 2)
