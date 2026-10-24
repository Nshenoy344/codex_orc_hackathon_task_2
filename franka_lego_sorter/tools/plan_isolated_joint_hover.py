"""Register measured trial obstacles and request PLAN ONLY; no action client use."""
import copy,hashlib,json,math,time
from pathlib import Path
import numpy as np,yaml
from geometry_msgs.msg import Pose
from shape_msgs.msg import SolidPrimitive,Plane
from moveit_msgs.msg import PlanningScene,CollisionObject,Constraints,PositionConstraint,OrientationConstraint,JointConstraint
from moveit_msgs.srv import ApplyPlanningScene,GetMotionPlan
from rosidl_runtime_py.convert import message_to_ordereddict
from scipy.spatial.transform import Rotation
from lego_sorter.cli import load_config
from lego_sorter.motion import RosRobot,LinearPlan
from lego_sorter.calibration_motion import slow_plan
from lego_sorter.isolated_brick import load_frame,_cloud
from lego_sorter.charuco_observation import fit_depth_plane

root=Path('data/commissioning');snapshot=root/'isolated_repositioned_view.npz'
f,g=load_frame(snapshot);he=yaml.safe_load((root/'charuco_rgbd_plane_candidate.yaml').read_text());bc=g@np.array(he['T_tcp_color'])
n,d,quality=fit_depth_plane(f,np.array([[550,450],[750,450],[750,680],[550,680]],float))
if n[2]>0:n=-n;d=-d
nb=bc[:3,:3]@n;db=d-nb@bc[:3,3]
if nb[2]<.98 or quality['rms_m']>.0015:raise RuntimeError('Unqualified measured table plane')
cloud,pixels=_cloud(f,stride=3);world=cloud@bc[:3,:3].T+bc[:3,3];height=world@nb+db
objects=[];table=CollisionObject();table.id='lego_trial_table';table.header.frame_id='fr3_link0';table.operation=CollisionObject.ADD
plane=Plane();plane.coef=list(map(float,[*nb,db]));table.planes=[plane];pose=Pose();pose.orientation.w=1.;table.plane_poses=[pose];objects.append(table)
for name,(x0,y0,x1,y1),extend in [('lego_trial_input',(570,0,1279,310),.20),('lego_trial_tray',(0,0,410,370),.12)]:
 mask=(pixels[:,0]>=x0)&(pixels[:,0]<=x1)&(pixels[:,1]>=y0)&(pixels[:,1]<=y1)&(height>.012)&(height<.25)
 pts=world[mask]
 if len(pts)<100:raise RuntimeError('Insufficient observed '+name)
 lo,hi=np.percentile(pts,[.5,99.5],axis=0);lo-=.008;hi+=.008
 # Cropped far edge is conservatively extended away from the robot.
 hi[0]=min(.85,hi[0]+extend);lo[2]=min(lo[2],-.045)
 obj=CollisionObject();obj.id=name;obj.header.frame_id='fr3_link0';obj.operation=CollisionObject.ADD
 shape=SolidPrimitive();shape.type=SolidPrimitive.BOX;shape.dimensions=list(map(float,hi-lo));p=Pose();p.orientation.w=1.;p.position.x,p.position.y,p.position.z=map(float,(lo+hi)/2)
 obj.primitives=[shape];obj.primitive_poses=[p];objects.append(obj)
robot=RosRobot(load_config('config/demo.yaml'))
try:
 end=time.monotonic()+6
 while True:
  try:start=robot.current_pose();break
  except RuntimeError:
   if time.monotonic()>end:raise
   time.sleep(.05)
 if np.linalg.norm(start[:3,3]-g[:3,3])>.001:raise RuntimeError('Snapshot no longer matches actual TCP')
 hand=robot.gripper_state()
 if abs(hand['width_m']-.030)>.0002:raise RuntimeError('Measured30mm trial opening changed')
 scene=PlanningScene();scene.is_diff=True;scene.robot_state.is_diff=True;scene.world.collision_objects=objects
 apply=robot._node.create_client(ApplyPlanningScene,'/apply_planning_scene')
 if not apply.wait_for_service(timeout_sec=6):raise RuntimeError('Scene service unavailable')
 ar=ApplyPlanningScene.Request();ar.scene=scene
 if not robot._wait(apply.call_async(ar),8).success:raise RuntimeError('Measured scene was not applied')
 target=np.array(json.loads((root/'isolated_repositioned_targets.json').read_text())['poses']['first_hover200'])
 yaw=math.atan2(start[1,0],start[0,0]);c,s=math.cos(yaw),math.sin(yaw);target[:3,:3]=[[c,s,0],[s,-c,0],[0,0,-1]]
 request=GetMotionPlan.Request();q=request.motion_plan_request;q.group_name=robot.cfg['planning_group'];q.num_planning_attempts=5;q.allowed_planning_time=10.;q.max_velocity_scaling_factor=.07;q.max_acceleration_scaling_factor=.05
 joint=copy.deepcopy(robot._fresh_arm_state()[0]);joint.name=list(joint.name);joint.position=list(joint.position)
 for name,value in hand['joints'].items():
  if name not in joint.name:joint.name.append(name);joint.position.append(value)
 q.start_state.joint_state=joint;q.start_state.is_diff=True
 p=Pose();p.position.x,p.position.y,p.position.z=map(float,target[:3,3]);p.orientation.x,p.orientation.y,p.orientation.z,p.orientation.w=map(float,Rotation.from_matrix(target[:3,:3]).as_quat())
 pc=PositionConstraint();pc.header.frame_id=robot.cfg['base_frame'];pc.link_name=robot.cfg['tcp_frame'];pc.weight=1.;sp=SolidPrimitive();sp.type=SolidPrimitive.SPHERE;sp.dimensions=[.0005];pc.constraint_region.primitives=[sp];pc.constraint_region.primitive_poses=[p]
 oc=OrientationConstraint();oc.header.frame_id=robot.cfg['base_frame'];oc.link_name=robot.cfg['tcp_frame'];oc.orientation=p.orientation;oc.absolute_x_axis_tolerance=.001;oc.absolute_y_axis_tolerance=.001;oc.absolute_z_axis_tolerance=.001;oc.weight=1.
 co=Constraints();co.position_constraints=[pc];co.orientation_constraints=[oc];q.goal_constraints=[co]
 for name,position,below,above in [('fr3_joint5',0.,1.5,1.5),('fr3_joint6',1.75,1.25,1.25)]:
  bound=JointConstraint();bound.joint_name=name;bound.position=position;bound.tolerance_below=below;bound.tolerance_above=above;bound.weight=1.;q.path_constraints.joint_constraints.append(bound)
 path_pc=PositionConstraint();path_pc.header.frame_id=robot.cfg['base_frame'];path_pc.link_name=robot.cfg['tcp_frame'];path_pc.weight=1.
 region=SolidPrimitive();region.type=SolidPrimitive.BOX;region.dimensions=[.75,.80,.45]
 region_pose=Pose();region_pose.orientation.w=1.;region_pose.position.x=.275;region_pose.position.z=.375
 path_pc.constraint_region.primitives=[region];path_pc.constraint_region.primitive_poses=[region_pose];q.path_constraints.position_constraints=[path_pc]
 client=robot._node.create_client(GetMotionPlan,'/plan_kinematic_path')
 if not client.wait_for_service(timeout_sec=6):raise RuntimeError('Plan-only service unavailable')
 response=robot._wait(client.call_async(request),20).motion_plan_response
 if response.error_code.val!=1:raise RuntimeError('Plan-only error '+str(response.error_code.val))
 plan=slow_plan(LinearPlan(start,target,tuple(joint.name),tuple(joint.position),response.trajectory),minimum_duration_sec=15)
 from lego_sorter.application_joint_bounds import validate_application_joint_bounds
 validate_application_joint_bounds(plan.trajectory)
 data=plan.to_dict();data['trajectory']=message_to_ordereddict(plan.trajectory)
 data.update(base_frame=robot.cfg['base_frame'],tcp_frame=robot.cfg['tcp_frame'],isolated_observation={'purpose':'noncontact_hover200','planning_method':'joint_space_plan_only','snapshot':str(snapshot.resolve()),'snapshot_sha256':hashlib.sha256(snapshot.read_bytes()).hexdigest(),'geometry_report':str((root/'isolated_repositioned_geometry.json').resolve()),'gripper_width_m':hand['width_m'],'table_plane_base':[ *nb.tolist(),float(db)],'registered_scene':message_to_ordereddict(scene),'action_sent':False})
 out=root/'isolated_hover200_joint_high_plan.json'
 if out.exists():raise RuntimeError('Existing reviewable plan will not be overwritten')
 out.write_text(json.dumps(data,indent=2));print(json.dumps({'plan':str(out),'duration_sec':data['duration_sec'],'T_base_tcp_target':target.tolist(),'action_sent':False},indent=2))
finally:robot.close()
