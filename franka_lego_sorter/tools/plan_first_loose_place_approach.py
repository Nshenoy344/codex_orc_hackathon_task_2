"""Plan only exact first loose-placement approach; no action or finger commands."""
import argparse,json,time,hashlib
from pathlib import Path
import numpy as np
from rosidl_runtime_py.convert import message_to_ordereddict
from lego_sorter.cli import load_config
from lego_sorter.motion import RosRobot
from lego_sorter.calibration_motion import slow_plan
from lego_sorter.held_view_checkpoint import verify_checkpoint
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',required=True);p.add_argument('--target',default='data/commissioning/first_loose_placement_target.json');p.add_argument('--snapshot',default='data/commissioning/template_reference_uncovered_paired.npz');p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output)
if out.exists():raise RuntimeError('Existing plan cannot be overwritten')
c=verify_checkpoint(a.checkpoint);target=json.loads(Path(a.target).read_text())
if target['kind']!='first_loose_placement_target_only' or target['production_placement_ready']is not False:raise ValueError('Explicit unqualified placement diagnostic required')
for item in target['source_files']:
 if hashlib.sha256(Path(item['path']).read_bytes()).hexdigest()!=item['sha256']:raise ValueError('Placement source changed')
r=RosRobot(load_config('config/demo.yaml'))
try:
 end=time.monotonic()+6
 while True:
  try:start=r.current_pose();break
  except RuntimeError:
   if time.monotonic()>end:raise
   time.sleep(.05)
 width=float(r.gripper_state()['width_m']);verify_checkpoint(a.checkpoint,start,width)
 plan=slow_plan(r.plan_linear(np.array(target['T_base_tcp_approach30'])),minimum_duration_sec=15.)
 report=plan.to_dict();report['trajectory']=message_to_ordereddict(plan.trajectory)
 meta=dict(stage='first_loose_place_approach',checkpoint=str(Path(a.checkpoint).resolve()),placement_target=str(Path(a.target).resolve()),scene_snapshot=str(Path(a.snapshot).resolve()),handeye=c['handeye'],environment_boxes=c['environment_boxes'],gripper_width_m=width,gripper_action_authorized=False,production_sorting_transfer=False,action_sent=False)
 for key in ('checkpoint','placement_target','scene_snapshot','handeye','environment_boxes'):meta[key+'_sha256']=hashlib.sha256(Path(meta[key]).read_bytes()).hexdigest()
 report.update(base_frame=r.cfg['base_frame'],tcp_frame=r.cfg['tcp_frame'],isolated_diagnostic_stage=meta);out.write_text(json.dumps(report,indent=2));print(json.dumps(dict(plan=str(out),action_sent=False)))
finally:r.close()
