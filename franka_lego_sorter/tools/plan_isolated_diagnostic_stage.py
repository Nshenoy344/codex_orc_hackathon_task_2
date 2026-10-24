"""Plan only an explicitly enabled isolated-brick diagnostic stage."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
from rosidl_runtime_py.convert import message_to_ordereddict
from lego_sorter.cli import load_config
from lego_sorter.motion import RosRobot
from lego_sorter.calibration_motion import slow_plan
p=argparse.ArgumentParser();p.add_argument('--stage',choices=['approach30'],required=True);p.add_argument('--targets',required=True);p.add_argument('--plane',choices=['left','right'],default='left');p.add_argument('--snapshot',required=True);p.add_argument('--handeye',required=True);p.add_argument('--output',required=True);a=p.parse_args()
targets=Path(a.targets).resolve();snapshot=Path(a.snapshot).resolve();handeye=Path(a.handeye).resolve();data=json.loads(targets.read_text());plane=next(x for x in data['planes']if x['plane']==a.plane);target=np.array(plane['poses']['hover30'],float);g=np.load(snapshot)['T_base_tcp'];out=Path(a.output)
if out.exists():raise RuntimeError('Existing stage plan cannot be overwritten')
r=RosRobot(load_config('config/demo.yaml'))
try:
 deadline=time.monotonic()+6
 while True:
  try:start=r.current_pose();break
  except RuntimeError:
   if time.monotonic()>deadline:raise
   time.sleep(.05)
 # A static world-anchored target remains valid after small holding drift.
 # The execution adapter independently retains its1mm fresh-trajectory-start gate.
 if np.linalg.norm(start[:3,3]-g[:3,3])>.005:raise RuntimeError('Approach source observation outside5mm acquisition envelope')
 opening=float(r.gripper_state()['width_m']);plan=slow_plan(r.plan_linear(target),minimum_duration_sec=12.)
 report=plan.to_dict();report['trajectory']=message_to_ordereddict(plan.trajectory);report.update(base_frame=r.cfg['base_frame'],tcp_frame=r.cfg['tcp_frame'],isolated_diagnostic_stage={'stage':a.stage,'targets':str(targets),'plane':a.plane,'snapshot':str(snapshot),'handeye':str(handeye),'gripper_width_m':opening,'physical_contact_authorized':False,'action_sent':False})
 out.write_text(json.dumps(report,indent=2));print(json.dumps({'plan':str(out),'stage':a.stage,'T_base_tcp_target':target.tolist(),'opening_m':opening,'action_sent':False},indent=2))
finally:r.close()
