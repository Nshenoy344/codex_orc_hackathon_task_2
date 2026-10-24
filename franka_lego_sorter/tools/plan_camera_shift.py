"""Plan only a bounded horizontal viewing shift from a measured RGBD snapshot."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np,yaml
from rosidl_runtime_py.convert import message_to_ordereddict
from lego_sorter.cli import load_config
from lego_sorter.motion import RosRobot
from lego_sorter.calibration_motion import slow_plan
p=argparse.ArgumentParser();p.add_argument('--snapshot',required=True);p.add_argument('--handeye',required=True);p.add_argument('--output',required=True);p.add_argument('--distance-m',type=float,default=.060);a=p.parse_args()
if not 0<a.distance_m<=.060:raise ValueError('Viewing step must be positive and at most60mm')
snapshot=Path(a.snapshot).resolve();handeye=Path(a.handeye).resolve();record=np.load(snapshot);measured=record['T_base_tcp'];X=np.array(yaml.safe_load(handeye.read_text())['T_tcp_color']);out=Path(a.output)
if out.exists():raise RuntimeError('Existing plan must not be overwritten or replayed')
r=RosRobot(load_config('config/demo.yaml'))
try:
 deadline=time.monotonic()+6
 while True:
  try:start=r.current_pose();break
  except RuntimeError:
   if time.monotonic()>deadline:raise
   time.sleep(.05)
 angle=np.arccos(np.clip((np.trace(start[:3,:3].T@measured[:3,:3])-1)/2,-1,1))
 if np.linalg.norm(start[:3,3]-measured[:3,3])>.001 or angle>.01:raise RuntimeError('Measured snapshot no longer matches actual TCP')
 direction=(start@X)[:3,1].copy();direction[2]=0.;direction/=np.linalg.norm(direction)
 target=start.copy();target[:3,3]+=direction*a.distance_m
 opening=float(r.gripper_state()['width_m']);plan=slow_plan(r.plan_linear(target),minimum_duration_sec=8.)
 data=plan.to_dict();data['trajectory']=message_to_ordereddict(plan.trajectory)
 data.update(base_frame=r.cfg['base_frame'],tcp_frame=r.cfg['tcp_frame'],commissioning_view_shift=dict(purpose='bounded_horizontal_camera_reframing',camera_axis='positive_y_projected_xy',distance_m=a.distance_m,direction_base=direction.tolist(),snapshot=str(snapshot),snapshot_sha256=hashlib.sha256(snapshot.read_bytes()).hexdigest(),handeye=str(handeye),handeye_sha256=hashlib.sha256(handeye.read_bytes()).hexdigest(),gripper_width_m=opening,action_sent=False))
 out.write_text(json.dumps(data,indent=2));print(json.dumps({'plan':str(out),'T_base_tcp_target':target.tolist(),'measured_opening_m':opening,'action_sent':False},indent=2))
finally:r.close()
