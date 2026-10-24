#!/usr/bin/env bash
# Source this file; it does not launch devices, nodes or robot commands.
reference_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source /opt/ros/humble/setup.bash
if [[ -f "$reference_root/sources/robot_ws/install/setup.bash" ]]; then
  source "$reference_root/sources/robot_ws/install/setup.bash"
fi
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1 ROS_DOMAIN_ID="${LEGO_ROS_DOMAIN_ID:-31}"
export PYTHONPATH="$reference_root/sources/franka_lego_sorter:$reference_root/sources/franka_realsense_handeye_ros2:$reference_root/sources/pyrealsense2_runtime:${PYTHONPATH:-}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
