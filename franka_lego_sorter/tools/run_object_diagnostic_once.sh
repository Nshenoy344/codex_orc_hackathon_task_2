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
target="${3:-data/commissioning/misplaced_maroon_recovery_target.json}"
stage="${4:-approach30}"
[[ "$prefix" =~ ^[a-zA-Z0-9_]+$ ]] || exit 2
[[ "$mode" == --plan-only || "$mode" == --execute ]] || exit 2
plan="data/commissioning/${prefix}_plan.json"
result="data/commissioning/${prefix}_result.json"
[[ ! -e "$plan" && ! -e "$result" ]] || { echo 'Existing attempt; never replay' >&2; exit 2; }
args=(--target "$target" --stage "$stage" --output "$plan")
if [[ "$stage" == grasp_pose ]]; then
  args+=(--reference-plan "${5:?approach plan}" --reference-result "${6:?approach result}")
fi
/usr/bin/python3 tools/plan_object_diagnostic.py "${args[@]}" > "logs/${prefix}_planning.log" 2>&1
/usr/bin/python3 tools/validate_object_diagnostic.py "$plan" > "logs/${prefix}_validation.log" 2>&1
if [[ "$mode" == --execute ]]; then
  bash tools/commission.sh execute-plan --plan-file "$plan" --output "$result" > "logs/${prefix}_execution.log" 2>&1
else
  echo "Plan and validation saved: $plan (no motion sent)"
fi
