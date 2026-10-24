"""Read-only MoveIt IK probe for an explicitly saved TCP pose; sends no action."""
import argparse,json
from pathlib import Path
from scipy.spatial.transform import Rotation
from moveit_msgs.srv import GetPositionIK
from lego_sorter.cli import load_config
from lego_sorter.geometry import matrix
from lego_sorter.motion import RosRobot

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('target_json');p.add_argument('--key',default='poses.grasp')
p.add_argument('--config',default='config/demo.yaml');p.add_argument('--output',required=True)
a=p.parse_args();data=json.loads(Path(a.target_json).read_text());target=data
for key in a.key.split('.'):target=target[key]
target=matrix(target);robot=RosRobot(load_config(a.config))
try:
 client=robot._node.create_client(GetPositionIK,'/compute_ik')
 if not client.wait_for_service(timeout_sec=6):raise RuntimeError('MoveIt IK service unavailable')
 req=GetPositionIK.Request();q=req.ik_request
 q.group_name=robot.cfg['planning_group'];q.ik_link_name=robot.cfg['tcp_frame']
 q.robot_state.joint_state=robot._fresh_arm_state()[0];q.robot_state.is_diff=True
 q.pose_stamped.header.frame_id=robot.cfg['base_frame'];q.avoid_collisions=True;q.timeout.sec=2
 q.pose_stamped.pose.position.x,q.pose_stamped.pose.position.y,q.pose_stamped.pose.position.z=map(float,target[:3,3])
 quat=Rotation.from_matrix(target[:3,:3]).as_quat()
 q.pose_stamped.pose.orientation.x,q.pose_stamped.pose.orientation.y,q.pose_stamped.pose.orientation.z,q.pose_stamped.pose.orientation.w=map(float,quat)
 r=robot._wait(client.call_async(req),8)
 report=dict(read_only=True,action_sent=False,target_source=str(Path(a.target_json).resolve()),pose_key=a.key,T_base_tcp=target.tolist(),error_code=int(r.error_code.val),ik_found=r.error_code.val==1,joint_names=list(r.solution.joint_state.name),joint_positions=list(r.solution.joint_state.position),limitations='IK/self collision against currently registered MoveIt scene only; observed scene and tool geometry require separate review.')
 Path(a.output).write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
finally:robot.close()
