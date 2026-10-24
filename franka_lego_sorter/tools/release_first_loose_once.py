"""One diagnostic20mm native Hand opening; no arm action and no occupancy commit."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
from lego_sorter.endpoint_servo import pose_error


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def check_pose_and_hold(actual,width,target):
 actual=np.asarray(actual,float)
 if actual.shape!=(4,4)or not np.isfinite(actual).all():raise ValueError('Invalid fresh physical pose')
 goal=np.asarray(target['T_base_tcp_release'],float);delta,angle=pose_error(goal,actual)
 if np.linalg.norm(delta)>.001 or angle>.01:raise ValueError('Fresh physical release endpoint outside1mm/.01rad; no opening')
 if not np.isfinite(width)or abs(width-.01535)>.0002:raise ValueError('Fresh heldbody width outside15.35±.2mm; no opening')
 normal=np.asarray(target['support_plane_base']['normal'],float);normal/=np.linalg.norm(normal)
 offset=float(target['support_plane_base']['offset_p99_m'])+float(target['minimum_clearance_above_p99_m'])-float(target['minimum_clearance_above_observed_max_m'])
 bottom=np.asarray(target['estimated_bottom_footprint_tcp_m'],float)@actual[:3,:3].T+actual[:3,3];clearance=float((bottom@normal-offset).min())
 if clearance<.001:raise ValueError('Fresh conditional heldbottom less than1mm above actual slotmax; no opening')
 return dict(physical_position_error_m=float(np.linalg.norm(delta)),physical_rotation_error_rad=float(angle),minimum_bottom_clearance_above_slot_observed_max_m=clearance)

def validate_evidence(plan_path,result_path):
 plan=json.loads(Path(plan_path).read_text());result=json.loads(Path(result_path).read_text());meta=plan['isolated_diagnostic_stage'];inner=result.get('first_loose_release_pose_result',{})
 if meta.get('stage')!='first_loose_release_pose'or result.get('success')is not True or inner.get('success')is not True or inner.get('holding_width_verified')is not True or Path(result['plan_file']).resolve()!=Path(plan_path).resolve():raise ValueError('Successful matching strict release-pose and verified heldwidth required')
 for key,limit in [('physical_position_error_m',.001),('physical_rotation_error_rad',.01)]:
  value=float(inner.get(key,float('inf')))
  if not np.isfinite(value)or value>limit:raise ValueError('Release-pose endpoint proof outside strictbound')
 clearance=float(inner.get('minimum_bottom_clearance_above_slot_observed_max_m',-float('inf')))
 if not np.isfinite(clearance)or clearance<.001:raise ValueError('Release-pose proof lacks1mm slotclearance')
 target_path=Path(meta['targets']).resolve()
 if digest(target_path)!=meta['targets_sha256']:raise ValueError('Placement target changed')
 target=json.loads(target_path.read_text())
 if target.get('kind')!='first_loose_placement_target_only'or target.get('production_placement_ready')is not False or not np.allclose(meta['physical_goal'],target['T_base_tcp_release'],atol=1e-9):raise ValueError('Wrong diagnostic physical target')
 for item in target['source_files']:
  if digest(item['path'])!=item['sha256']:raise ValueError('Pinned placement evidence changed')
 review_path=Path(plan_path).with_suffix('.validation.json');review=json.loads(review_path.read_text())
 if review.get('passed_numeric_checks')is not True or review.get('purpose')!='isolated_first_loose_release_pose'or review.get('plan_sha256')!=digest(plan_path):raise ValueError('Matching passed release-pose review required')
 return target_path,target,review_path

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',required=True);p.add_argument('--pose-result',required=True);p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output).resolve()
 if out.exists():raise RuntimeError('Existing release attempt output; do not repeat')
 target_path,target,review_path=validate_evidence(a.plan,a.pose_result);marker=target_path.with_suffix('.hand_release_attempt.json')
 if marker.exists():raise RuntimeError('Prior release attempt exists for this target; no repeat under another output filename')
 from lego_sorter.cli import load_config
 from lego_sorter.motion import RosRobot
 r=RosRobot(load_config('config/demo.yaml'));record=dict(format='first_loose_hand_release_attempt_v1',success=False,status='prepared',arm_action_sent=False,production_counts_committed=False,occupancy_committed=False,calibration_verified=False,commanded_opening_m=.020)
 for key,path in [('pose_plan',a.plan),('pose_result',a.pose_result),('placement_target',target_path),('pose_review',review_path)]:record[key]=str(Path(path).resolve());record[key+'_sha256']=digest(path)
 def save():
  tmp=out.with_suffix(out.suffix+'.tmp');tmp.write_text(json.dumps(record,indent=2));tmp.replace(out)
 attempted=False
 try:
  end=time.monotonic()+6
  while True:
   try:actual=r.current_pose();break
   except RuntimeError:
    if time.monotonic()>end:raise
    time.sleep(.05)
  before=r.gripper_state();checks=check_pose_and_hold(actual,float(before['width_m']),target)
  record.update(status='release_action_pending',gripper_action_may_have_occurred=True,T_base_tcp_before=actual.tolist(),gripper_before=before,**checks)
  out.parent.mkdir(parents=True,exist_ok=True)
  # Both exclusivity records exist before the nativeHand command. Never retry
  # automatically if action delivery or post-action verification is uncertain.
  with marker.open('x') as stream:json.dump(dict(output=str(out),pose_result=str(Path(a.pose_result).resolve()),pose_result_sha256=digest(a.pose_result),status='release_action_pending'),stream,indent=2)
  with out.open('x') as stream:json.dump(record,stream,indent=2)
  attempted=True;r.open_gripper(width_m=.020)
  end=time.monotonic()+float(r.cfg['state_timeout_sec'])
  while True:
   after=r.gripper_state()
   if abs(float(after['width_m'])-.020)<=.0002:break
   if time.monotonic()>end:raise RuntimeError('Hand command returned but measured opening not20±.2mm')
   time.sleep(.03)
  record.update(success=True,status='released_pending_visual_verification',gripper_after=after,opening_width_verified=True,brick_placement_verified=False);save();print(json.dumps(record,indent=2))
 except BaseException as error:
  if attempted:record.update(success=False,status='release_failed_or_unverified',reason=str(error));save()
  raise
 finally:r.close()
if __name__=='__main__':main()
