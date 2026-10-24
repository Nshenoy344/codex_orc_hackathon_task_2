#!/usr/bin/env bash
set -eo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
source tools/analysis_limits.sh
source /opt/ros/humble/setup.bash
source /home/nshenoy/franka_ros2_ws/install/setup.bash
export PYTHONNOUSERSITE=1 ROS_DOMAIN_ID="${LEGO_ROS_DOMAIN_ID:-31}"
export PYTHONPATH="$PWD:$HOME/.local/share/franka-camera-viewer/python:${PYTHONPATH:-}"
set -u
prefix="${1:?unique attempt prefix}"
mode="${2:---plan-only}"
[[ "$prefix" =~ ^[a-zA-Z0-9_]+$ ]] || exit 2
[[ "$mode" == --plan-only || "$mode" == --execute ]] || exit 2
plan="data/commissioning/${prefix}_plan.json"
result="data/commissioning/${prefix}_result.json"
[[ ! -e "$plan" && ! -e "$result" ]] || { echo 'Existing attempt; never replay' >&2; exit 2; }
/usr/bin/python3 tools/plan_first_loose_release_pose.py --targets data/commissioning/first_loose_placement_target.json --snapshot data/commissioning/template_reference_uncovered_paired.npz --handeye data/commissioning/charuco_rgbd_plane_candidate.yaml --reference-plan data/commissioning/first_loose_approach_01_plan.json --reference-result data/commissioning/first_loose_approach_01_result.json --output "$plan" > "logs/${prefix}_planning.log" 2>&1
/usr/bin/python3 tools/validate_first_loose_release_pose.py "$plan" > "logs/${prefix}_validation.log" 2>&1
if [[ "$mode" == --execute ]]; then
  bash tools/commission.sh execute-plan --plan-file "$plan" --output "$result" > "logs/${prefix}_execution.log" 2>&1
else
  echo "Plan and validation saved: $plan (no motion sent)"
fi
