"""One verified isolated-brick Hand grasp; never moves the arm or lifts."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
from lego_sorter.cli import load_config
from lego_sorter.motion import RosRobot
from lego_sorter.endpoint_servo import pose_error
from lego_sorter.isolated_grasp_pose import verify_review

p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--pose-result',required=True);p.add_argument('--output',required=True);a=p.parse_args()
plan_path=Path(a.plan).resolve();result_path=Path(a.pose_result).resolve();out=Path(a.output)
if out.exists():raise RuntimeError('Existing Hand attempt: do not repeat without inspection')
plan=json.loads(plan_path.read_text());result=json.loads(result_path.read_text());review=json.loads(plan_path.with_suffix('.validation.json').read_text());verify_review(plan_path,review)
if result.get('success')is not True or Path(result['plan_file']).resolve()!=plan_path:raise RuntimeError('Successful matching grasp-pose execution required')
meta=plan['isolated_diagnostic_stage'];source=json.loads(Path(meta['targets']).read_text());plane=next(v for v in source['planes']if v['plane']==meta['plane']);goal=np.array(meta['physical_goal'])
r=RosRobot(load_config('config/demo.yaml'));record={'success':False,'status':'prepared','arm_action_sent':False,'lift_action_sent':False,'pose_plan':str(plan_path),'pose_result':str(result_path),'pose_plan_sha256':hashlib.sha256(plan_path.read_bytes()).hexdigest(),'pose_result_sha256':hashlib.sha256(result_path.read_bytes()).hexdigest(),'production_sorting_transfer':False}
def save():
 tmp=out.with_suffix(out.suffix+'.tmp');tmp.write_text(json.dumps(record,indent=2));tmp.replace(out)
try:
 deadline=time.monotonic()+6
 while True:
  try:actual=r.current_pose();break
  except RuntimeError:
   if time.monotonic()>deadline:raise
   time.sleep(.05)
 delta,angle=pose_error(goal,actual);normal=np.array(plane['measured_table_plane']['normal_base']);floor=np.array(plane['floor_ray_point_base_m']);tips=np.array([[x,y,.0095]for x in(-.011,.011)for y in(-.05,.05)]);clearance=float(((tips@actual[:3,:3].T+actual[:3,3]-floor)@normal).min())
 if np.linalg.norm(delta)>.001 or angle>.01 or clearance<.003:raise RuntimeError('Fresh pose/table clearance changed; no finger command sent')
 before=r.gripper_state()
 if abs(before['width_m']-.030)>.0002:raise RuntimeError('Fresh30mm preopen required')
 if not 0<float(r.cfg['grasp_force_n'])<=20:raise RuntimeError('Isolated diagnostic force must be at most20N')
 record.update(status='grasp_action_pending',gripper_action_may_have_occurred=True,commanded_width_m=.0158,commanded_force_n=float(r.cfg['grasp_force_n']),T_base_tcp_before=actual.tolist(),minimum_modeled_tip_floor_clearance_m=clearance,gripper_before=before);save()
 success=r.grasp(width_m=.0158)
 if not success:raise RuntimeError('Hand did not report a successful grasp; no lift sent')
 deadline=time.monotonic()+float(r.cfg['state_timeout_sec'])
 while True:
  after=r.gripper_state()
  if abs(after['width_m']-.0158)<=.001:break
  if time.monotonic()>deadline:raise RuntimeError('Successful action but measured opening does not match brick body; no lift sent')
  time.sleep(.03)
 record.update(success=True,status='grasped_pending_lift',gripper_after=after,hand_grasp_and_body_width_verified=True,brick_lifted_verified=False);save();print(json.dumps(record,indent=2))
except BaseException as e:
 record.update(success=False,status='failed_or_unverified',reason=str(e))
 try:record['gripper_after']=r.gripper_state()
 except Exception:pass
 save();raise
finally:r.close()
