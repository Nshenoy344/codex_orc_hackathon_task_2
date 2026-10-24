"""One explicit empty-space Hand move; no arm action."""
import argparse,json,time
from pathlib import Path
from lego_sorter.cli import load_config
from lego_sorter.motion import RosRobot
p=argparse.ArgumentParser();p.add_argument('--width-m',type=float,default=.030);p.add_argument('--minimum-tcp-z-m',type=float,default=.30);p.add_argument('--output',default='data/commissioning/trial_hand_preopen.json');a=p.parse_args()
if not .15<=a.minimum_tcp_z_m<=.4 or not .016<a.width_m<=.08:raise ValueError('Invalid explicit empty-space Hand bounds')
out=Path(a.output)
if out.exists():raise RuntimeError('Existing Hand outcome must not be overwritten')
r=RosRobot(load_config('config/demo.yaml'))
try:
 deadline=time.monotonic()+6
 while True:
  try:p=r.current_pose();break
  except RuntimeError:
   if time.monotonic()>deadline:raise
   time.sleep(.05)
 if p[2,3]<a.minimum_tcp_z_m:raise RuntimeError('Empty-space Hand test below explicit raised bound')
 before=r.gripper_state()
 r.open_gripper(width_m=a.width_m)
 deadline=time.monotonic()+float(r.cfg['state_timeout_sec'])
 while True:
  after=r.gripper_state()
  if abs(float(after['width_m'])-a.width_m)<=.0002:break
  if time.monotonic()>deadline:raise RuntimeError('Measured Hand opening did not settle within0.2mm of target')
  time.sleep(.03)
 report={'success':True,'arm_action_sent':False,'commanded_width_m':a.width_m,'width_verified_tolerance_m':.0002,'T_base_tcp':p.tolist(),'before':before,'after':after}
 out.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
except BaseException as e:
 out.write_text(json.dumps({'success':False,'reason':str(e),'arm_action_sent':False},indent=2));raise
finally:r.close()
