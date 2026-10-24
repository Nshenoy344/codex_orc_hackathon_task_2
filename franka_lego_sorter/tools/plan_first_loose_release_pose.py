"""Plan only the isolated nominal grasp pose with bounded measured tracking bias.

No opening or contact action; target-only conditional held geometry.
"""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from rosidl_runtime_py.convert import message_to_ordereddict
from lego_sorter.cli import load_config
from lego_sorter.motion import RosRobot
from lego_sorter.calibration_motion import slow_plan
p=argparse.ArgumentParser();p.add_argument('--targets',required=True);p.add_argument('--snapshot',required=True);p.add_argument('--handeye',required=True);p.add_argument('--reference-plan',required=True);p.add_argument('--reference-result',required=True);p.add_argument('--output',required=True);a=p.parse_args()
paths={k:Path(getattr(a,k.replace('-','_'))).resolve()for k in ['targets','snapshot','handeye','reference_plan','reference_result']}
t=json.loads(paths['targets'].read_text());physical=np.array(t['T_base_tcp_release'],float)
for item in t['source_files']:
 if hashlib.sha256(Path(item['path']).read_bytes()).hexdigest()!=item['sha256']:raise RuntimeError('Pinned placement source changed')
reference=json.loads(paths['reference_plan'].read_text());result=json.loads(paths['reference_result'].read_text());out=Path(a.output)
if result.get('success')is not True:raise RuntimeError('A successful approach reference is required')
if out.exists():raise RuntimeError('Existing grasp-pose plan will not be overwritten')
r=RosRobot(load_config('config/demo.yaml'))
try:
 deadline=time.monotonic()+6
 while True:
  try:start=r.current_pose();break
  except RuntimeError:
   if time.monotonic()>deadline:raise
   time.sleep(.05)
 goal=np.array(reference['T_base_tcp_target']);bias=goal[:3,3]-start[:3,3];errorR=goal[:3,:3]@start[:3,:3].T;angle=Rotation.from_matrix(errorR).magnitude()
 if np.linalg.norm(bias)>.003 or angle>.01:raise RuntimeError('Fresh approach tracking bias outside3mm/.01rad bound')
 opening=float(r.gripper_state()['width_m'])
 if abs(opening-.01535)>.0003:raise RuntimeError('Fresh held width differs from15.35mm by>.3mm')
 command=physical.copy();command[:3,3]+=bias;command[:3,:3]=errorR@physical[:3,:3]
 plan=slow_plan(r.plan_linear(command),minimum_duration_sec=12.)
 data=plan.to_dict();data['trajectory']=message_to_ordereddict(plan.trajectory);data.update(base_frame=r.cfg['base_frame'],tcp_frame=r.cfg['tcp_frame'],isolated_diagnostic_stage={'stage':'first_loose_release_pose','targets':str(paths['targets']),'snapshot':str(paths['snapshot']),'handeye':str(paths['handeye']),'gripper_width_m':opening,'reference_plan':str(paths['reference_plan']),'reference_result':str(paths['reference_result']),'physical_goal':physical.tolist(),'measured_tracking_bias_m':bias.tolist(),'measured_tracking_rotation_rad':float(angle),'gripper_action_authorized':False,'action_sent':False})
 
 for key in paths:data['isolated_diagnostic_stage'][key+'_sha256']=hashlib.sha256(paths[key].read_bytes()).hexdigest()
 out.write_text(json.dumps(data,indent=2));print(json.dumps({'plan':str(out),'physical_goal':physical.tolist(),'command_target':command.tolist(),'tracking_bias_m':bias.tolist(),'action_sent':False},indent=2))
finally:r.close()
