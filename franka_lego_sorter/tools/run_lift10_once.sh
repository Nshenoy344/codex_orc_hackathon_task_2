#!/usr/bin/env bash
set -eo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
source tools/analysis_limits.sh
source /opt/ros/humble/setup.bash
source /home/nshenoy/franka_ros2_ws/install/setup.bash
export PYTHONNOUSERSITE=1 ROS_DOMAIN_ID="${LEGO_ROS_DOMAIN_ID:-31}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
set -u
prefix="${1:?unique attempt prefix}"
[[ "$prefix" =~ ^[a-zA-Z0-9_]+$ ]] || exit 2
plan="data/commissioning/${prefix}_plan.json";result="data/commissioning/${prefix}_result.json"
[[ ! -e "$plan" && ! -e "$result" ]] || { echo 'Existing attempt; never replay' >&2; exit 2; }
/usr/bin/python3 tools/plan_isolated_lift10.py --hold-result data/commissioning/isolated_first_grasp.json --targets data/commissioning/isolated_close_nominal_diagnostic_targets.json --snapshot data/commissioning/isolated_close_observation.npz --handeye data/commissioning/charuco_rgbd_plane_candidate.yaml --output "$plan" > "logs/${prefix}_planning.log" 2>&1
/usr/bin/python3 tools/validate_isolated_lift10.py "$plan" > "logs/${prefix}_validation.log" 2>&1
bash tools/commission.sh execute-plan --plan-file "$plan" --output "$result" > "logs/${prefix}_execution.log" 2>&1
