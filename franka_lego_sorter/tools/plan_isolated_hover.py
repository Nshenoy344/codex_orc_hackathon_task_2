"""Plan only the initial raised observation; never sends an action."""
import json,math
from pathlib import Path
import numpy as np
from rosidl_runtime_py.convert import message_to_ordereddict
from lego_sorter.cli import load_config
from lego_sorter.motion import RosRobot
from lego_sorter.calibration_motion import slow_plan

root=Path('data/commissioning');targets=json.loads((root/'isolated_repositioned_targets.json').read_text())
robot=RosRobot(load_config('config/demo.yaml'))
try:
 import time
 end=time.monotonic()+6
 while True:
  try:start=robot.current_pose();break
  except RuntimeError:
   if time.monotonic()>=end:raise
   time.sleep(.05)
 target=np.array(targets['poses']['grasp'],float);target[2,3]+=.200
 yaw=math.atan2(start[1,0],start[0,0]);c,s=math.cos(yaw),math.sin(yaw)
 target[:3,:3]=[[c,s,0],[s,-c,0],[0,0,-1]]
 if not(.15<target[0,3]<.5 and abs(target[1,3])<.3 and .15<target[2,3]<.3):raise RuntimeError('Hover target outside raised initial trial region')
 plan=slow_plan(robot.plan_linear(target),minimum_duration_sec=12)
 data=plan.to_dict();data['trajectory']=message_to_ordereddict(plan.trajectory)
 data.update(base_frame=robot.cfg['base_frame'],tcp_frame=robot.cfg['tcp_frame'],isolated_observation={'purpose':'noncontact_hover200_preserve_yaw_vertical_tool','geometry_report':targets['geometry_report'],'minimum_tip_height_above_target_m':.15,'action_sent':False})
 path=root/'isolated_hover200_plan.json'
 if path.exists():raise RuntimeError('Existing plan must not be overwritten')
 path.write_text(json.dumps(data,indent=2));print(json.dumps({'saved':str(path),'T_base_tcp_start':start.tolist(),'T_base_tcp_target':target.tolist(),'fraction':plan.fraction},indent=2))
finally:robot.close()
