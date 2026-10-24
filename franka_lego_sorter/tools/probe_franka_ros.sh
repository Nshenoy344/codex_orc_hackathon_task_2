#!/usr/bin/env bash
# Read-only ROS graph, controller state, joint states and TF. Never launches robot control.
set -eo pipefail
source /opt/ros/humble/setup.bash
source /home/nshenoy/franka_ros2_ws/install/setup.bash
export PYTHONNOUSERSITE=1
export ROS_DOMAIN_ID="${LEGO_ROS_DOMAIN_ID:-31}"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec timeout 16 /usr/bin/python3 "$script_dir/probe_franka_ros.py"
