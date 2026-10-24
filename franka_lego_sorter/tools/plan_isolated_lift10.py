"""Plan only one10mm lift after an evidenced isolated Hand grasp."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
from rosidl_runtime_py.convert import message_to_ordereddict
from lego_sorter.cli import load_config
from lego_sorter.motion import RosRobot
from lego_sorter.calibration_motion import slow_plan
p=argparse.ArgumentParser();p.add_argument('--hold-result',required=True);p.add_argument('--targets',required=True);p.add_argument('--plane',default='left');p.add_argument('--snapshot',required=True);p.add_argument('--handeye',required=True);p.add_argument('--output',required=True);p.add_argument('--carry-lift10-result');a=p.parse_args()
paths={k:Path(getattr(a,k)).resolve()for k in['hold_result','targets','snapshot','handeye']};hold=json.loads(paths['hold_result'].read_text());out=Path(a.output)
if hold.get('success')is not True or hold.get('status')!='grasped_pending_lift':raise RuntimeError('Successful isolated holding evidence required')
if out.exists():raise RuntimeError('Existing lift attempt must not be overwritten')
r=RosRobot(load_config('config/demo.yaml'))
try:
 deadline=time.monotonic()+6
 while True:
  try:start=r.current_pose();break
  except RuntimeError:
   if time.monotonic()>deadline:raise
   time.sleep(.05)
 opening=float(r.gripper_state()['width_m'])
 if abs(opening-.0158)>.001:raise RuntimeError('Fresh opening does not match brick body')
 target=start.copy()
 if a.carry_lift10_result:
  checkpoint=json.loads(Path(a.carry_lift10_result).read_text())
  if checkpoint.get('success')is not True:raise RuntimeError('Successful10mm checkpoint required')
  target[2,3]=.300;stage='carry_observe300';minimum_duration=15.
 else:
  target[2,3]+=.010;stage='lift10';minimum_duration=12.
 plan=slow_plan(r.plan_linear(target),minimum_duration_sec=minimum_duration)
 report=plan.to_dict();report['trajectory']=message_to_ordereddict(plan.trajectory);report.update(base_frame=r.cfg['base_frame'],tcp_frame=r.cfg['tcp_frame'],isolated_diagnostic_stage={'stage':stage,'hold_result':str(paths['hold_result']),'targets':str(paths['targets']),'plane':a.plane,'snapshot':str(paths['snapshot']),'handeye':str(paths['handeye']),'gripper_width_m':opening,'source_roi_px':[690,480,110,160],'gripper_action_authorized':False,'production_sorting_transfer':False,'action_sent':False})
 if a.carry_lift10_result:
  lift_path=Path(a.carry_lift10_result).resolve()
  report['isolated_diagnostic_stage']['lift10_result']=str(lift_path)
  report['isolated_diagnostic_stage']['holding_checkpoint']={'T_base_tcp':start.tolist(),'width_m':opening,'lift10_result_sha256':hashlib.sha256(lift_path.read_bytes()).hexdigest(),'method':'fresh ROS TCP and Hand width; no arm action sent for checkpoint'}
 out.write_text(json.dumps(report,indent=2));print(json.dumps({'plan':str(out),'target':target.tolist(),'holding_width_m':opening,'action_sent':False},indent=2))
finally:r.close()
