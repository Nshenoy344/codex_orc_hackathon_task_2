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
mkdir -p logs data/commissioning
checkpoint="data/commissioning/${prefix}_checkpoint.json"
plan="data/commissioning/${prefix}_plan.json"
result="data/commissioning/${prefix}_result.json"
[[ ! -e "$checkpoint" && ! -e "$plan" && ! -e "$result" ]] || { echo 'Existing attempt; never replay' >&2; exit 2; }
/usr/bin/python3 tools/capture_held_view_checkpoint.py --snapshot data/commissioning/template_reference_uncovered_paired.npz --result data/commissioning/template_view_02_result.json --output "$checkpoint" > "logs/${prefix}_checkpoint.log" 2>&1
/usr/bin/python3 tools/plan_first_loose_place_approach.py --checkpoint "$checkpoint" --output "$plan" > "logs/${prefix}_planning.log" 2>&1
/usr/bin/python3 tools/validate_first_loose_place_approach.py "$plan" > "logs/${prefix}_validation.log" 2>&1
if [[ "$mode" == --execute ]]; then
  bash tools/commission.sh execute-plan --plan-file "$plan" --output "$result" > "logs/${prefix}_execution.log" 2>&1
else
  echo "Plan and validation saved: $plan (no motion sent)"
fi
