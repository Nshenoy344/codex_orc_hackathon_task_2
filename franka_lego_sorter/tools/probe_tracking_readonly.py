#!/usr/bin/env python3
"""Bounded read-only tracking/controller/robot-state sample. Sends no goals or writes."""
import json,time,argparse
from pathlib import Path
import rclpy
from rclpy.qos import qos_profile_sensor_data
from control_msgs.msg import JointTrajectoryControllerState
from franka_msgs.msg import FrankaRobotState
from rosidl_runtime_py.convert import message_to_ordereddict

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);args=parser.parse_args()
    rclpy.init();node=rclpy.create_node('lego_tracking_readonly_probe')
    samples={'controller':[],'robot':[]}
    def callback(kind):
        def receive(msg):
            samples[kind].append(message_to_ordereddict(msg))
        return receive
    s1=node.create_subscription(JointTrajectoryControllerState,'/fr3_arm_controller/controller_state',callback('controller'),qos_profile_sensor_data)
    s2=node.create_subscription(FrankaRobotState,'/franka_robot_state_broadcaster/robot_state',callback('robot'),qos_profile_sensor_data)
    try:
        deadline=time.monotonic()+6.
        while time.monotonic()<deadline:rclpy.spin_once(node,timeout_sec=.1)
        out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
        out.write_text(json.dumps(samples,indent=2))
        print(json.dumps({'output':str(out),'controller_samples':len(samples['controller']),'robot_samples':len(samples['robot'])}))
    finally:node.destroy_node();rclpy.shutdown()
if __name__=='__main__':main()
