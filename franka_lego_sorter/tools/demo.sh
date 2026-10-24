#!/usr/bin/env bash
set -eo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/analysis_limits.sh"
sorter_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONNOUSERSITE=1
source /opt/ros/humble/setup.bash
source /home/nshenoy/franka_ros2_ws/install/setup.bash
export ROS_DOMAIN_ID="${LEGO_ROS_DOMAIN_ID:-31}"
export PYTHONPATH="$sorter_root:$HOME/.local/share/franka-camera-viewer/python:${PYTHONPATH:-}"
cd "$sorter_root"
mkdir -p logs
exec > >(tee -a "$sorter_root/logs/demo.log") 2>&1
if [[ "${1:-}" == camera ]]; then
    shift
    exec /usr/bin/python3 -u -m lego_sorter.perception_cli "$@" --config config/demo.yaml
fi
exec /usr/bin/python3 -u -m lego_sorter.cli "$@"
