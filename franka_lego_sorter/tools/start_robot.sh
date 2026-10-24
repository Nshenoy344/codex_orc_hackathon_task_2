#!/usr/bin/env bash
set -eo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/analysis_limits.sh"
source /opt/ros/humble/setup.bash
source /home/nshenoy/franka_ros2_ws/install/setup.bash
export ROS_DOMAIN_ID="${LEGO_ROS_DOMAIN_ID:-31}"
exec ros2 launch franka_fr3_moveit_config moveit.launch.py robot_ip:=172.16.0.2 use_fake_hardware:=false
