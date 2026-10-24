#!/usr/bin/env bash
set -eo pipefail
root_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$root_dir/tools/analysis_limits.sh"
source /opt/ros/humble/setup.bash
source /home/nshenoy/franka_ros2_ws/install/setup.bash
export PYTHONNOUSERSITE=1 ROS_DOMAIN_ID="${LEGO_ROS_DOMAIN_ID:-31}" PYTHONPATH="$root_dir:${PYTHONPATH:-}"
cd "$root_dir"
exec /usr/bin/python3 tools/probe_target_ik.py "$@"
