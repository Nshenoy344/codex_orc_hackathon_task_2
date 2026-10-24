"""Plan only a200mm isolated-brick observation from explicit measured artifacts."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
from rosidl_runtime_py.convert import message_to_ordereddict
from lego_sorter.cli import load_config
from lego_sorter.motion import RosRobot
from lego_sorter.calibration_motion import slow_plan
p=argparse.ArgumentParser();p.add_argument('--snapshot',required=True);p.add_argument('--targets',required=True);p.add_argument('--output',required=True);a=p.parse_args()
snapshot=Path(a.snapshot).resolve();targets=Path(a.targets).resolve();saved=np.load(snapshot);g=saved['T_base_tcp'];t=json.loads(targets.read_text());out=Path(a.output)
if out.exists():raise RuntimeError('Existing observation plan must not be overwritten')
target=np.array(t['poses']['grasp'],float);target[2,3]+=.2
r=RosRobot(load_config('config/demo.yaml'))
try:
 deadline=time.monotonic()+6
 while True:
  try:start=r.current_pose();break
  except RuntimeError:
   if time.monotonic()>deadline:raise
   time.sleep(.05)
 angle=np.arccos(np.clip((np.trace(start[:3,:3].T@g[:3,:3])-1)/2,-1,1))
 if np.linalg.norm(start[:3,3]-g[:3,3])>.001 or angle>.01:raise RuntimeError('Observation snapshot no longer matches actual TCP')
 opening=float(r.gripper_state()['width_m']);plan=slow_plan(r.plan_linear(target),minimum_duration_sec=12.)
 data=plan.to_dict();data['trajectory']=message_to_ordereddict(plan.trajectory)
 data.update(base_frame=r.cfg['base_frame'],tcp_frame=r.cfg['tcp_frame'],isolated_observation={'purpose':'noncontact_hover200','planning_method':'cartesian_plan_only','snapshot':str(snapshot),'snapshot_sha256':hashlib.sha256(snapshot.read_bytes()).hexdigest(),'geometry_report':str(Path(t['geometry_report']).resolve()),'gripper_width_m':opening,'physical_contact_authorized':False,'action_sent':False})
 out.write_text(json.dumps(data,indent=2));print(json.dumps({'plan':str(out),'duration_sec':data['duration_sec'],'T_base_tcp_target':target.tolist(),'opening_m':opening,'action_sent':False},indent=2))
finally:r.close()
