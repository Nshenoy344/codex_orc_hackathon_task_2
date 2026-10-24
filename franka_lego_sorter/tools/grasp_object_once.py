"""One body closure for explicitly tracked misplaced-brick recovery; no arm/lift."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
from lego_sorter.endpoint_servo import pose_error
from lego_sorter.object_diagnostic import verify_review

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def fresh_checks(actual,width,goal,target):
 actual=np.asarray(actual,float);goal=np.asarray(goal,float)
 if actual.shape!=(4,4)or not np.isfinite(actual).all():raise ValueError('Invalid fresh TCP')
 delta,angle=pose_error(goal,actual)
 if np.linalg.norm(delta)>.001 or angle>.01:raise ValueError('Fresh physical goal outside1mm/.01rad')
 if not np.isfinite(width)or abs(width-.030)>.0002:raise ValueError('Fresh30mm preopen±.2mm required')
 n=np.asarray(target['normal_base'],float);lower=float(target['visible_upper_p99_offset_base_m'])-float(target['toe_center_insertion_below_observed_p99_m'])-float(target['minimum_toe_clearance_above_prior_lower_layer_globalmax_m']);toe=np.array([[x,y,.0095]for x in(-.011,.011)for y in(-.05,.05)]);clearance=float(((toe@actual[:3,:3].T+actual[:3,3])@n-lower).min())
 if not np.isfinite(clearance)or clearance<.003:raise ValueError('Fresh fulltoe less than3mm above prior lowerlayer')
 return dict(physical_position_error_m=float(np.linalg.norm(delta)),physical_rotation_error_rad=float(angle),minimum_toe_above_prior_lower_layer_m=clearance)

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--pose-plan',required=True);p.add_argument('--pose-result',required=True);p.add_argument('--output',required=True);a=p.parse_args();planpath=Path(a.pose_plan).resolve();resultpath=Path(a.pose_result).resolve();out=Path(a.output).resolve();marker=planpath.with_suffix('.hand_grasp_attempt.json')
 if out.exists()or marker.exists():raise RuntimeError('Existing closure attempt; inspect rather than replay')
 plan=json.loads(planpath.read_text());m=plan['isolated_diagnostic_stage'];result=json.loads(resultpath.read_text());inner=result.get('object_diagnostic_result',{});reviewpath=planpath.with_suffix('.validation.json');verify_review(planpath,json.loads(reviewpath.read_text()))
 if m.get('stage')!='object_grasp_pose'or result.get('success')is not True or Path(result['plan_file']).resolve()!=planpath:raise ValueError('Successful matching object_grasp_pose required')
 for field,threshold in [('physical_position_error_m',.001),('physical_rotation_error_rad',.01)]:
  value=float(inner.get(field,float('inf')))
  if not np.isfinite(value)or value>threshold:raise ValueError('Strict object physical endpoint proof missing')
 clearance=float(inner.get('minimum_toe_above_prior_lower_layer_m',-float('inf')))
 if not np.isfinite(clearance)or clearance<.003:raise ValueError('Pose result lacks3mm lowerlayer clearance')
 targetpath=Path(m['object_target']);target=json.loads(targetpath.read_text());goal=np.array(m['physical_goal'])
 from lego_sorter.cli import load_config
 from lego_sorter.motion import RosRobot
 r=RosRobot(load_config('config/demo.yaml'));record=dict(format='tracked_object_hand_grasp_once_v1',success=False,status='prepared',arm_action_sent=False,lift_action_sent=False,production_counts_committed=False,occupancy_committed=False,automatic_color_classification=False,identity_context='Recovery of uniquely tracked operator-confirmed maroon brick; closure is not new color evidence',commanded_width_m=.0158,commanded_force_n=15.)
 for key,path in [('pose_plan',planpath),('pose_result',resultpath),('pose_review',reviewpath),('object_target',targetpath)]:record[key]=str(Path(path).resolve());record[key+'_sha256']=digest(path)
 attempted=False
 def save():
  temp=out.with_suffix(out.suffix+'.tmp');temp.write_text(json.dumps(record,indent=2));temp.replace(out)
 try:
  if not np.isclose(float(r.cfg['grasp_force_n']),15.,rtol=0,atol=1e-9):raise RuntimeError('Recovery requires configured15N graspforce')
  deadline=time.monotonic()+6
  while True:
   try:actual=r.current_pose();break
   except RuntimeError:
    if time.monotonic()>deadline:raise
    time.sleep(.05)
  before=r.gripper_state();checks=fresh_checks(actual,float(before['width_m']),goal,target)
  record.update(status='grasp_action_pending',gripper_action_may_have_occurred=True,T_base_tcp_before=actual.tolist(),gripper_before=before,**checks);out.parent.mkdir(parents=True,exist_ok=True)
  with marker.open('x')as stream:json.dump(dict(output=str(out),pose_result=str(resultpath),pose_result_sha256=digest(resultpath),status='grasp_action_pending'),stream,indent=2)
  with out.open('x')as stream:json.dump(record,stream,indent=2)
  attempted=True;success=r.grasp(width_m=.0158);record['sdk_grasp_success']=bool(success)
  if not success:raise RuntimeError('SDK grasp not successful; no lift or retry')
  deadline=time.monotonic()+float(r.cfg['state_timeout_sec'])
  while True:
   after=r.gripper_state()
   if abs(float(after['width_m'])-.01535)<=.001:break
   if time.monotonic()>deadline:raise RuntimeError('Measured bodywidth outside15.35±1mm; no lift')
   time.sleep(.03)
  record.update(success=True,status='grasped_pending_lift',gripper_after=after,hand_grasp_and_body_width_verified=True,brick_lifted_verified=False);save();print(json.dumps(record,indent=2))
 except BaseException as error:
  if attempted:
   record.update(success=False,status='failed_or_unverified_inspect',reason=str(error))
   try:record['gripper_after']=r.gripper_state()
   except Exception:pass
   save()
  raise
 finally:r.close()
if __name__=='__main__':main()
