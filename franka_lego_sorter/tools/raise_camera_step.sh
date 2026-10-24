#!/usr/bin/env bash
# One fixed-orientation +20mm base-Z commissioning step. Never moves fingers.
set -euo pipefail
root_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root_dir"
step_id="${1:?supply a unique step id}"
if [[ ! "$step_id" =~ ^[a-zA-Z0-9_]+$ ]]; then exit 2; fi
plan_path="data/commissioning/${step_id}_plan.json"
report_path="data/commissioning/${step_id}_validation.json"
result_path="data/commissioning/${step_id}_result.json"
if [[ -e "$plan_path" || -e "$result_path" ]]; then
  echo 'Step artifact already exists; do not repeat an executed step' >&2
  exit 2
fi
bash tools/commission.sh plan-lift --output "$plan_path" > "logs/${step_id}_planning.log" 2>&1
PYTHONNOUSERSITE=1 /usr/bin/python3 tools/validate_lift_plan.py "$plan_path" \
  --urdf data/commissioning/fr3_stock_hand.urdf --output "$report_path" > "logs/${step_id}_validation.log" 2>&1
# The independent validator must pass before the execution command is reached.
if ! bash tools/commission.sh execute-plan --plan-file "$plan_path" --output "$result_path"; then
  # A successful controller result can still fail measured endpoint verification.
  # Never repeat the20mm travel. Only that specific failure may get one bounded
  # feedback correction toward the SAME physical goal.
  PYTHONNOUSERSITE=1 /usr/bin/python3 - "$result_path" <<'CHECK'
import json,sys
r=json.load(open(sys.argv[1]))
if not str(r.get('reason','')).startswith('Pose verification failed:'):
    raise SystemExit('No automatic correction of action/communication/fault failures')
if r.get('target_position_error_m',1.)>.003:
    raise SystemExit('Observed endpoint error exceeds3mm correction envelope')
CHECK
  correction="data/commissioning/${step_id}_correction.json"
  bash tools/commission.sh plan-correction --plan-file "$plan_path" --output "$correction" > "logs/${step_id}_correction_planning.log" 2>&1
  PYTHONNOUSERSITE=1 /usr/bin/python3 tools/validate_endpoint_correction.py "$correction" \
    --urdf data/commissioning/fr3_stock_hand.urdf --output "${correction%.json}.validation.json" > "logs/${step_id}_correction_validation.log" 2>&1
  bash tools/commission.sh execute-plan --plan-file "$correction" --output "data/commissioning/${step_id}_correction_result.json"
fi
bash tools/commission.sh record --output "data/commissioning/${step_id}_view.npz"
