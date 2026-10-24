"""Plan only an explicit measured-object diagnostic stage; no Hand action."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
from rosidl_runtime_py.convert import message_to_ordereddict
from lego_sorter.cli import load_config
from lego_sorter.motion import RosRobot
from lego_sorter.calibration_motion import slow_plan


def main():
 p=argparse.ArgumentParser();p.add_argument('--target',required=True);p.add_argument('--stage',choices=['approach30','grasp_pose'],default='approach30');p.add_argument('--environment-boxes',default='data/commissioning/template_view_environment.json');p.add_argument('--output',required=True);p.add_argument('--reference-plan');p.add_argument('--reference-result');a=p.parse_args();target_path=Path(a.target).resolve();target=json.loads(target_path.read_text());out=Path(a.output)
 if out.exists():raise RuntimeError('Existing object attempt must not be overwritten')
 if target.get('production_ready')is not False or target.get('kind')!='isolated_misplaced_brick_recovery_target_only':raise RuntimeError('Explicit measured diagnostic object target required')
 for source in target['source_files']:
  if hashlib.sha256(Path(source['path']).read_bytes()).hexdigest()!=source['sha256']:raise RuntimeError('Measured object evidence changed')
 handeye=next(Path(v['path']).resolve()for v in target['source_files']if Path(v['path']).suffix=='.yaml');snapshot=Path(target['snapshot']).resolve();r=RosRobot(load_config('config/demo.yaml'))
 try:
  end=time.monotonic()+6
  while True:
   try:start=r.current_pose();break
   except RuntimeError:
    if time.monotonic()>end:raise
    time.sleep(.05)
  opening=float(r.gripper_state()['width_m'])
  if abs(opening-float(target['gripper_opening_m']))>.0002:raise RuntimeError('Fresh30mm preopen required before object approach')
  goal=np.array(target['T_base_tcp_approach30']);extra={};duration=15.
  if a.stage=='grasp_pose':
   if not a.reference_plan or not a.reference_result:raise RuntimeError('Measured approach plan/result required for grasp-pose bias')
   from scipy.spatial.transform import Rotation
   refpath=Path(a.reference_plan).resolve();respath=Path(a.reference_result).resolve();ref=json.loads(refpath.read_text());res=json.loads(respath.read_text());allowed=res.get('success')is True or(res.get('success')is False and res.get('status')=='execution_or_verification_failed'and res.get('reason','').startswith('Pose verification failed:'))
   if not allowed or Path(res['plan_file']).resolve()!=refpath:raise RuntimeError('Only completed approach or preserved endpoint-verification failure is a bias reference')
   approach=np.array(ref['T_base_tcp_target']);delta=approach[:3,3]-start[:3,3];turn=approach[:3,:3]@start[:3,:3].T
   from lego_sorter.object_diagnostic import bounded_approach_rotation
   if np.linalg.norm(delta)>.003:raise RuntimeError('Fresh local approach translation bias exceeds3mm')
   turn,rotation_diagnostics=bounded_approach_rotation(turn)
   physical=np.array(target['T_base_tcp_grasp']);goal=physical.copy();goal[:3,3]+=delta;goal[:3,:3]=turn@physical[:3,:3];duration=12.
   extra=dict(reference_plan=str(refpath),reference_result=str(respath),reference_execution_success=bool(res.get('success')),physical_goal=physical.tolist(),measured_approach_bias_m=delta.tolist(),**rotation_diagnostics)
  plan=slow_plan(r.plan_linear(goal),minimum_duration_sec=duration);data=plan.to_dict();data['trajectory']=message_to_ordereddict(plan.trajectory)
  metadata=dict(stage='object_'+a.stage,object_target=str(target_path),scene_snapshot=str(snapshot),handeye=str(handeye),environment_boxes=str(Path(a.environment_boxes).resolve()),gripper_width_m=opening,gripper_action_authorized=False,production_sorting_transfer=False,action_sent=False)
  metadata.update(extra)
  data.update(base_frame=r.cfg['base_frame'],tcp_frame=r.cfg['tcp_frame'],isolated_diagnostic_stage=metadata);out.write_text(json.dumps(data,indent=2));print(json.dumps({'plan':str(out),'action_sent':False}))
 finally:r.close()

if __name__=='__main__':main()
