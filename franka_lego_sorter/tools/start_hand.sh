#!/usr/bin/env bash
# Connect/publish Hand state only; no homing, opening or grasp commands on startup.
set -eo pipefail
source /opt/ros/humble/setup.bash
source /home/nshenoy/franka_ros2_ws/install/setup.bash
export ROS_DOMAIN_ID="${LEGO_ROS_DOMAIN_ID:-31}"
exec ros2 launch franka_gripper gripper.launch.py robot_ip:=172.16.0.2 namespace:=/
