#!/usr/bin/env bash
set -eo pipefail
source /opt/ros/humble/setup.bash
source /home/nshenoy/franka_ros2_ws/install/setup.bash
export ROS_DOMAIN_ID=31 PYTHONNOUSERSITE=1
exec /usr/bin/python3 "$(dirname -- "${BASH_SOURCE[0]}")/read_ros_snapshot.py"
