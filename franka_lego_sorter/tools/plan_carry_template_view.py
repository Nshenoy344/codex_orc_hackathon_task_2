"""PLAN ONLY: register measured obstacles/nominal held brick, then Cartesian view plan.

This script sends no trajectory, arm action, Hand command or camera request.
Planning-scene updates are reversible scene-model changes only.
"""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
from geometry_msgs.msg import Pose
from shape_msgs.msg import Plane,SolidPrimitive
from moveit_msgs.msg import PlanningScene,CollisionObject,AttachedCollisionObject,PlanningSceneComponents
from moveit_msgs.srv import ApplyPlanningScene,GetPlanningScene
from rosidl_runtime_py.convert import message_to_ordereddict
from rosidl_runtime_py.set_message import set_message_fields
from lego_sorter.cli import load_config
from lego_sorter.motion import RosRobot
from lego_sorter.calibration_motion import slow_plan
from lego_sorter.geometry import matrix
from lego_sorter.endpoint_servo import pose_error
from lego_sorter.environment_boxes import load_boxes


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True)
    defaults={'hold_result':'isolated_first_grasp.json','targets':'isolated_close_nominal_diagnostic_targets.json','snapshot':'isolated_close_observation.npz','handeye':'charuco_rgbd_plane_candidate.yaml','carry_result':'isolated_carry_observe300_checkpoint_result.json','source_proof':'isolated_first_pick_source_proof.json','scene_snapshot':'isolated_source_after_pick.npz','view_hint':'template_reference_view_hint.json','environment_boxes':'template_view_environment.json'}
    for key,value in defaults.items():p.add_argument('--'+key.replace('_','-'),default='data/commissioning/'+value)
    p.add_argument('--plane',default='left',choices=['left','right']);a=p.parse_args();paths={key:Path(getattr(a,key)).resolve()for key in defaults};out=Path(a.output)
    if out.exists():raise RuntimeError('Existing attempt must not be overwritten/replayed')
    hold=json.loads(paths['hold_result'].read_text());carry=json.loads(paths['carry_result'].read_text());proof=json.loads(paths['source_proof'].read_text());environment=json.loads(paths['environment_boxes'].read_text());hint=json.loads(paths['view_hint'].read_text());source=json.loads(paths['targets'].read_text());floor=next(v for v in source['planes']if v['plane']==a.plane)
    if hold.get('success')is not True or carry.get('success')is not True or proof.get('combined_pick_evidence')is not True:raise RuntimeError('Successful hold, carry and source-pick proof required')
    if environment['source_scene_sha256']!=hashlib.sha256(Path(environment['source_scene_artifact']).read_bytes()).hexdigest():raise RuntimeError('Recorded input-pallet source changed')
    load_boxes(paths['environment_boxes']);target=matrix(hint['poses']['template_reference_view_hint'])
    r=RosRobot(load_config('config/demo.yaml'))
    try:
        deadline=time.monotonic()+6
        while True:
            try:start=r.current_pose();break
            except RuntimeError:
                if time.monotonic()>deadline:raise
                time.sleep(.05)
        opening=float(r.gripper_state()['width_m']);delta,angle=pose_error(np.array(carry['T_base_tcp_after']),start)
        if np.linalg.norm(delta)>.003 or angle>.01 or abs(opening-float(hold['gripper_after']['width_m']))>.0002:raise RuntimeError('Current holding checkpoint differs from successful carry')
        if np.linalg.norm(target[:3,3]-start[:3,3])>.400 or target[2,3]<.300:raise RuntimeError('Viewing hint exceeds bounded viewing displacement')
        scene=PlanningScene();scene.is_diff=True;scene.robot_state.is_diff=True
        for saved in environment['collision_objects']:
            obj=CollisionObject();fields=dict(saved);fields.pop('operation',None);set_message_fields(obj,fields);obj.operation=CollisionObject.ADD;scene.world.collision_objects.append(obj)
        table=CollisionObject();table.id='lego_carried_view_table';table.header.frame_id=r.cfg['base_frame'];table.operation=CollisionObject.ADD
        n=np.array(floor['measured_table_plane']['normal_base']);fp=np.array(floor['floor_ray_point_base_m']);pl=Plane();pl.coef=[*map(float,n),float(-n@fp)];table.planes=[pl];pose=Pose();pose.orientation.w=1.;table.plane_poses=[pose];scene.world.collision_objects.append(table)
        gp=json.loads(Path(hold['pose_plan']).read_text());physical=matrix(gp['isolated_diagnostic_stage']['physical_goal']);center=np.array(floor['assumed_nominal_stud_top_base_m'])-n*.0057;local=(center-physical[:3,3])@physical[:3,:3]
        attached=AttachedCollisionObject();attached.link_name=r.cfg['tcp_frame'];attached.touch_links=['fr3_hand','fr3_leftfinger','fr3_rightfinger'];obj=attached.object;obj.id='lego_isolated_held_nominal';obj.header.frame_id=r.cfg['tcp_frame'];obj.operation=CollisionObject.ADD
        primitive=SolidPrimitive();primitive.type=SolidPrimitive.BOX;primitive.dimensions=[.0321,.0158,.0114];pose=Pose();pose.orientation.w=1.;pose.position.x,pose.position.y,pose.position.z=map(float,local);obj.primitives=[primitive];obj.primitive_poses=[pose];scene.robot_state.attached_collision_objects=[attached]
        client=r._node.create_client(ApplyPlanningScene,'/apply_planning_scene')
        if not client.wait_for_service(timeout_sec=6):raise RuntimeError('Apply planning-scene service unavailable')
        request=ApplyPlanningScene.Request();request.scene=scene
        if not r._wait(client.call_async(request),8).success:raise RuntimeError('Measured obstacles/nominal held object were not applied')
        getter=r._node.create_client(GetPlanningScene,'/get_planning_scene')
        if not getter.wait_for_service(timeout_sec=6):raise RuntimeError('Readback planning-scene service unavailable')
        query=GetPlanningScene.Request();query.components.components=PlanningSceneComponents.WORLD_OBJECT_GEOMETRY|PlanningSceneComponents.ROBOT_STATE_ATTACHED_OBJECTS
        readback=r._wait(getter.call_async(query),8).scene
        if not {v.id for v in scene.world.collision_objects}.issubset({v.id for v in readback.world.collision_objects})or obj.id not in {v.object.id for v in readback.robot_state.attached_collision_objects}:raise RuntimeError('Scene readback lacks expected obstacles or held box')
        plan=slow_plan(r.plan_linear(target),minimum_duration_sec=15.)
        report=plan.to_dict();report['trajectory']=message_to_ordereddict(plan.trajectory)
        metadata={key:str(value)for key,value in paths.items()};metadata.update(stage='carry_template_view',plane=a.plane,gripper_width_m=opening,source_roi_px=[690,480,110,160],holding_checkpoint={'T_base_tcp':start.tolist(),'width_m':opening,'carry_result_sha256':hashlib.sha256(paths['carry_result'].read_bytes()).hexdigest(),'method':'Fresh ROS TCP/Hand checkpoint; no motion action'},gripper_action_authorized=False,production_sorting_transfer=False,action_sent=False)
        report.update(base_frame=r.cfg['base_frame'],tcp_frame=r.cfg['tcp_frame'],isolated_diagnostic_stage=metadata,planning_scene_applied=message_to_ordereddict(scene),planning_scene_readback_verified=True,attached_box_is_nominal=True)
        out.write_text(json.dumps(report,indent=2));print(json.dumps({'plan':str(out),'target':target.tolist(),'duration_sec':report['duration_sec'],'holding_width_m':opening,'planning_scene_readback_verified':True,'arm_action_sent':False,'hand_action_sent':False},indent=2))
    finally:r.close()

if __name__=='__main__':main()
