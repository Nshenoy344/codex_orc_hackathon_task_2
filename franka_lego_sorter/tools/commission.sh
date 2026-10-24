#!/usr/bin/env bash
set -eo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/analysis_limits.sh"
sorter_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONNOUSERSITE=1
source /opt/ros/humble/setup.bash
source /home/nshenoy/franka_ros2_ws/install/setup.bash
# Deliberate deployment domain, independent of unrelated shells using domain 23.
export ROS_DOMAIN_ID="${LEGO_ROS_DOMAIN_ID:-31}"
export PYTHONPATH="$sorter_root:$HOME/.local/share/franka-camera-viewer/python:${PYTHONPATH:-}"
cd "$sorter_root"
mkdir -p logs
exec > >(tee -a logs/commissioning.log) 2>&1
exec /usr/bin/python3 -u -m lego_sorter.commissioning "$@"
