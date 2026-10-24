#!/usr/bin/env bash
set -eo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
source tools/analysis_limits.sh
source /opt/ros/humble/setup.bash
source /home/nshenoy/franka_ros2_ws/install/setup.bash
export PYTHONNOUSERSITE=1 ROS_DOMAIN_ID=31
export PYTHONPATH="$PWD:$HOME/.local/share/franka-camera-viewer/python:${PYTHONPATH:-}"
exec /usr/bin/python3 -u tools/wrist_calibration_step.py "$@"
