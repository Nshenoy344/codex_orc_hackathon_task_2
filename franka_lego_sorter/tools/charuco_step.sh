#!/usr/bin/env bash
# One reviewed raised wrist rotation + measured ChArUco observation. No fingers.
set -euo pipefail
root_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root_dir"
step_id="${1:?unique id}";axis="${2:?x or y}";angle="${3:?absolute reference-relative degrees}"
reference="${4:?reference measured snapshot}";current="${5:?current measured snapshot}"
[[ "$step_id" =~ ^[a-zA-Z0-9_]+$ ]] || exit 2
plan="data/commissioning/${step_id}_plan.json"
result="data/commissioning/${step_id}_result.json"
[[ ! -e "$plan" && ! -e "$result" ]] || { echo 'Existing artifact: never repeat automatically' >&2; exit 2; }
bash tools/calibration_plan.sh plan --axis "$axis" --degrees "$angle" --max-step-deg 5.1 --max-center-drift-m .010 \
  --reference "$reference" --current "$current" --max-board-rms-px .8 --board-config data/commissioning/charuco_center_inspection.json --urdf data/commissioning/fr3_stock_hand.urdf \
  --output "$plan" > "logs/${step_id}_planning.log" 2>&1
if ! bash tools/commission.sh execute-plan --plan-file "$plan" --output "$result" > "logs/${step_id}_execution.log" 2>&1; then
  PYTHONNOUSERSITE=1 /usr/bin/python3 - "$result" <<'CHECK'
import json,sys
r=json.load(open(sys.argv[1]))
if not str(r.get('reason','')).startswith('Pose verification failed:'):
 raise SystemExit('Action/fault/communication failure: stop without another command')
CHECK
fi
set +e
PYTHONNOUSERSITE=1 /usr/bin/python3 - "$plan" "$result" <<'NEEDS'
import json,sys,numpy as np
p=json.load(open(sys.argv[1]));r=json.load(open(sys.argv[2]))
a=np.array(r['T_base_tcp_after']);g=np.array(p['T_base_tcp_target'])
e=float(np.linalg.norm(a[:3,3]-g[:3,3]))
angle=float(np.arccos(np.clip((np.trace(g[:3,:3].T@a[:3,:3])-1)/2,-1,1)))
if e>.003 or angle>.01:
 print('Measured pose error exceeds the bounded correction envelope',file=sys.stderr)
 raise SystemExit(2)
raise SystemExit(0 if e>.001 else 1)
NEEDS
correction_status=$?
set -e
if [[ "$correction_status" -gt 1 ]]; then exit "$correction_status"; fi
if [[ "$correction_status" -eq 0 ]]; then
 correction="data/commissioning/${step_id}_correction.json"
 bash tools/commission.sh plan-correction --plan-file "$plan" --output "$correction" > "logs/${step_id}_correction_planning.log" 2>&1
 PYTHONNOUSERSITE=1 /usr/bin/python3 tools/validate_endpoint_correction.py "$correction" \
   --urdf data/commissioning/fr3_stock_hand.urdf --output "${correction%.json}.validation.json" > "logs/${step_id}_correction_validation.log" 2>&1
 bash tools/commission.sh execute-plan --plan-file "$correction" --output "data/commissioning/${step_id}_correction_result.json" > "logs/${step_id}_correction_execution.log" 2>&1
fi
bash tools/commission.sh record --output "data/commissioning/${step_id}_view.npz" > "logs/${step_id}_capture.log" 2>&1
PYTHONNOUSERSITE=1 PYTHONPATH="$root_dir" /usr/bin/python3 -m lego_sorter.charuco_handeye \
 "data/commissioning/${step_id}_view.npz" --board-config data/commissioning/charuco_center_inspection.json \
 --output "data/commissioning/${step_id}_board.yaml" --max-rms-px .8 --inspect-only > "logs/${step_id}_board.log" 2>&1
printf 'Measured and checked observation: %s\n' "$step_id"
