"""One bounded, independently checked acquisition movement; never moves fingers."""
import argparse,json,time,sys
from pathlib import Path
import numpy as np,yaml
from scipy.spatial.transform import Rotation
from scipy.spatial import cKDTree
from rosidl_runtime_py.convert import message_to_ordereddict
from lego_sorter.motion import RosRobot
from lego_sorter.cli import load_config
from lego_sorter.calibration_motion import slow_plan
from lego_sorter.isolated_brick import load_frame,_cloud
from lego_sorter.charuco_observation import fit_depth_plane
from lego_sorter.charuco_handeye import observe_board
sys.path.insert(0,str(Path(__file__).resolve().parent))
from validate_isolated_hover import review

p=argparse.ArgumentParser();p.add_argument('--id',required=True);p.add_argument('--session',default='data/wrist_calibration_final');p.add_argument('--rx',type=float,default=0);p.add_argument('--ry',type=float,default=0);p.add_argument('--dx',type=float,default=0);p.add_argument('--dy',type=float,default=0);p.add_argument('--dz',type=float,default=.04);p.add_argument('--execute',action='store_true');a=p.parse_args()
if not a.id.replace('_','').isalnum():raise ValueError('Invalid unique step ID')
if max(abs(a.rx),abs(a.ry))>15 or np.hypot(a.rx,a.ry)>20 or max(abs(a.dx),abs(a.dy))>.03 or not 0<=a.dz<=.06:raise ValueError('Acquisition target outside bounded region')
folder=Path(a.session);prefix=folder/a.id;planfile=prefix.with_suffix('.plan.json');resultfile=prefix.with_suffix('.result.json')
if planfile.exists()or resultfile.exists():raise RuntimeError('Existing step; never replay')
sample=json.loads(sorted(folder.glob('sample_*.json'))[0].read_text());scene=Path(sample['rgbd_file']);frame,tcp=load_frame(scene);handeye=Path('data/commissioning/charuco_rgbd_plane_candidate.yaml');X=np.array(yaml.safe_load(handeye.read_text())['T_tcp_color']);B=tcp@X
board=yaml.safe_load(Path('data/commissioning/charuco_reference_board.yaml').read_text())['board_config'];observed=observe_board(frame,board,max_rms_px=.8,estimator='rgbd_plane');CB=np.array(observed['T_color_board']);n=B[:3,:3]@CB[:3,2]
if n[2]<0:n=-n
point=(B@CB)[:3,3];geometry=folder/'acquisition_scene_geometry.json'
geometry.write_text(json.dumps(dict(visible_top_base_m=point.tolist(),table_point_base_m=point.tolist(),measured_table_plane=dict(normal_base=n.tolist()),source_snapshot=str(scene),semantics='Current identified ChArUco board plane; fixed board during autonomous acquisition'),indent=2))
r=RosRobot(load_config('config/demo.yaml'));sent=False
try:
 end=time.monotonic()+6
 while True:
  try:start=r.current_pose();break
  except RuntimeError:
   if time.monotonic()>end:raise
   time.sleep(.05)
 referencefile=folder/'motion_reference.json'
 if referencefile.exists():reference=np.array(json.loads(referencefile.read_text())['T_base_tcp'])
 else:reference=start.copy();referencefile.write_text(json.dumps(dict(T_base_tcp=reference.tolist(),scope='Camera acquisition only; no landmark registration'),indent=2))
 target=reference.copy();target[:3,3]+=np.array([a.dx,a.dy,a.dz]);target[:3,:3]=Rotation.from_rotvec(np.radians([a.rx,a.ry,0.])).as_matrix()@reference[:3,:3]
 if np.linalg.norm(target[:3,3]-start[:3,3])>.07 or Rotation.from_matrix(start[:3,:3].T@target[:3,:3]).magnitude()>np.radians(12.1):raise ValueError('Individual step exceeds70mm/12degrees')
 opening=float(r.gripper_state()['width_m']);plan=slow_plan(r.plan_linear(target),minimum_duration_sec=8.)
 data=plan.to_dict();data['trajectory']=message_to_ordereddict(plan.trajectory);data.update(base_frame=r.cfg['base_frame'],tcp_frame=r.cfg['tcp_frame'],purpose='camera_calibration_acquisition_only',no_gripper_action=True)
 planfile.write_text(json.dumps(data,indent=2));reviewed=review(planfile,'data/commissioning/fr3_stock_hand.urdf',scene,handeye,geometry,joint_space=True,opening=opening,camera_mode='oriented_contact',include_dense_poses=True)
 poses=np.array(reviewed.pop('_dense_tcp_poses'));cloud,_=_cloud(frame,3);world=cloud@B[:3,:3].T+B[:3,3]
 # Conservative union for stock carrier + replacement tips + wrist webcam.
 # It extends beyond the CAD toe16.55mm; this is a noncontact acquisition
 # envelope only and does not certify a new-tip contact/grasp model.
 low=np.array([-.060,-.120,-.140]);high=np.array([.060,.120,.030]);corners=np.array([[x,y,z]for x in(low[0],high[0])for y in(low[1],high[1])for z in(low[2],high[2])]);minimum=10.;hits=0
 for T in poses:
  v=corners@T[:3,:3].T+T[:3,3];minimum=min(minimum,float(((v-point)@n).min()));local=(world-T[:3,3])@T[:3,:3];hits=max(hits,int(np.all((local>=low)&(local<=high),axis=1).sum()))
 reviewed['checks'].update(noncontact_complete_hand_and_webcam_union_clear=bool(hits==0),noncontact_union_board_clearance=bool(minimum>=.15))
 reviewed.update(scope='Noncontact camera acquisition only',additional_envelope_tcp_lower_m=low.tolist(),additional_envelope_tcp_upper_m=high.tolist(),minimum_additional_envelope_board_clearance_m=minimum,additional_envelope_maximum_cloud_hits=hits)
 reviewed['failed_checks']=[k for k,v in reviewed['checks'].items()if not v];reviewed['passed_numeric_checks']=not reviewed['failed_checks'];prefix.with_suffix('.validation.json').write_text(json.dumps(reviewed,indent=2))
 if reviewed['failed_checks']:raise RuntimeError('Acquisition path refused: '+str(reviewed['failed_checks']))
 if a.execute:
  # A5mm viewing tolerance is specific to this high-clearance acquisition.
  # Contact/grasp verification and the shared1mm fresh-start guard are unchanged.
  r.cfg['position_tolerance_m']=.005
  sent=True;actual=r.execute_linear_plan(plan);after=float(r.gripper_state()['width_m'])
  if abs(after-opening)>.0002:raise RuntimeError('Hand width changed during camera acquisition')
  result=dict(success=True,T_base_tcp_after=actual.tolist(),T_base_tcp_target=target.tolist(),position_error_m=float(np.linalg.norm(actual[:3,3]-target[:3,3])),gripper_action_sent=False,scope='Camera calibration only; landmark registration not started')
  resultfile.write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
 else:print(json.dumps(dict(plan=str(planfile),passed=True,action_sent=False)))
except BaseException as e:
 resultfile.write_text(json.dumps(dict(success=False,reason=str(e),physical_motion_may_have_occurred=sent),indent=2));raise
finally:r.close()
