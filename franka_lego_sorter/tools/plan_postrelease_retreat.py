"""Plan only vertical postrelease retreat; detach held model and add static proxy."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from geometry_msgs.msg import Pose
from shape_msgs.msg import SolidPrimitive
from moveit_msgs.msg import PlanningScene,CollisionObject,AttachedCollisionObject,PlanningSceneComponents
from moveit_msgs.srv import ApplyPlanningScene,GetPlanningScene
from rosidl_runtime_py.convert import message_to_ordereddict
from rosidl_runtime_py.set_message import set_message_fields
from lego_sorter.cli import load_config
from lego_sorter.motion import RosRobot
from lego_sorter.calibration_motion import slow_plan
from lego_sorter.endpoint_servo import pose_error
from lego_sorter.released_brick_geometry import static_bounds


def main():
 p=argparse.ArgumentParser();p.add_argument('--release-result',required=True);p.add_argument('--output',required=True);a=p.parse_args();release_path=Path(a.release_result).resolve();released=json.loads(release_path.read_text());out=Path(a.output)
 if out.exists():raise RuntimeError('Existing retreat attempt must not be overwritten')
 if released.get('success')is not True or released.get('opening_width_verified')is not True or released.get('status')!='released_pending_visual_verification':raise RuntimeError('Verified20mm release opening required')
 for key in('pose_plan','pose_result','placement_target','pose_review'):
  if hashlib.sha256(Path(released[key]).read_bytes()).hexdigest()!=released[key+'_sha256']:raise RuntimeError('Release evidence changed')
 target=json.loads(Path(released['placement_target']).read_text());poseplan=json.loads(Path(released['pose_plan']).read_text());previous=json.loads(Path(released['pose_review']).read_text());anchor,lo,hi=static_bounds(released,target)
 r=RosRobot(load_config('config/demo.yaml'))
 try:
  end=time.monotonic()+6
  while True:
   try:start=r.current_pose();break
   except RuntimeError:
    if time.monotonic()>end:raise
    time.sleep(.05)
  opening=float(r.gripper_state()['width_m']);delta,angle=pose_error(anchor,start)
  if np.linalg.norm(delta)>.003 or angle>.01 or abs(opening-.020)>.0002:raise RuntimeError('Fresh release pose/opening changed; no retreat command')
  goal=start.copy();goal[2,3]=.350
  if not 0<goal[2,3]-start[2,3]<=.350:raise RuntimeError('Retreat must be upward≤350mm')
  scene=PlanningScene();scene.is_diff=True;scene.robot_state.is_diff=True
  detach=AttachedCollisionObject();detach.link_name=r.cfg['tcp_frame'];detach.object.id='lego_isolated_held_nominal';detach.object.operation=CollisionObject.REMOVE;scene.robot_state.attached_collision_objects=[detach]
  for saved in json.loads(Path(previous['environment_boxes']).read_text())['collision_objects']:
   obj=CollisionObject();fields=dict(saved);fields.pop('operation',None);set_message_fields(obj,fields);obj.operation=CollisionObject.ADD;scene.world.collision_objects.append(obj)
  obj=CollisionObject();obj.id='lego_first_loose_released_static';obj.header.frame_id=r.cfg['base_frame'];obj.operation=CollisionObject.ADD;shape=SolidPrimitive();shape.type=SolidPrimitive.BOX;shape.dimensions=list(map(float,hi-lo));pose=Pose();position=anchor[:3,:3]@((lo+hi)/2)+anchor[:3,3];pose.position.x,pose.position.y,pose.position.z=map(float,position);pose.orientation.x,pose.orientation.y,pose.orientation.z,pose.orientation.w=map(float,Rotation.from_matrix(anchor[:3,:3]).as_quat());obj.primitives=[shape];obj.primitive_poses=[pose];scene.world.collision_objects.append(obj)
  client=r._node.create_client(ApplyPlanningScene,'/apply_planning_scene')
  if not client.wait_for_service(timeout_sec=6):raise RuntimeError('Scene update unavailable')
  req=ApplyPlanningScene.Request();req.scene=scene
  if not r._wait(client.call_async(req),8).success:raise RuntimeError('Static released model/detachment not applied')
  getter=r._node.create_client(GetPlanningScene,'/get_planning_scene')
  if not getter.wait_for_service(timeout_sec=6):raise RuntimeError('Scene readback unavailable')
  query=GetPlanningScene.Request();query.components.components=PlanningSceneComponents.WORLD_OBJECT_NAMES|PlanningSceneComponents.ROBOT_STATE_ATTACHED_OBJECTS;readback=r._wait(getter.call_async(query),8).scene
  if obj.id not in{v.id for v in readback.world.collision_objects}or any(v.object.id=='lego_isolated_held_nominal'for v in readback.robot_state.attached_collision_objects):raise RuntimeError('Static model or nominal detachment readback failed')
  plan=slow_plan(r.plan_linear(goal),minimum_duration_sec=15.);report=plan.to_dict();report['trajectory']=message_to_ordereddict(plan.trajectory)
  meta=dict(stage='postrelease_retreat350',release_result=str(release_path),placement_target=str(Path(released['placement_target']).resolve()),scene_snapshot=str(Path(poseplan['isolated_diagnostic_stage']['snapshot']).resolve()),handeye=str(Path(poseplan['isolated_diagnostic_stage']['handeye']).resolve()),environment_boxes=str(Path(previous['environment_boxes']).resolve()),gripper_width_m=opening,gripper_action_authorized=False,production_sorting_transfer=False,attached_model_removed=True,static_model_conditional=True,release_anchor_acquisition_delta_m=delta.tolist(),release_anchor_acquisition_distance_m=float(np.linalg.norm(delta)),release_anchor_acquisition_rotation_rad=float(angle),action_sent=False)
  report.update(base_frame=r.cfg['base_frame'],tcp_frame=r.cfg['tcp_frame'],isolated_diagnostic_stage=meta,planning_scene_applied=message_to_ordereddict(scene),planning_scene_readback_verified=True);out.write_text(json.dumps(report,indent=2));print(json.dumps({'plan':str(out),'action_sent':False,'attached_model_removed':True}))
 finally:r.close()

if __name__=='__main__':main()
