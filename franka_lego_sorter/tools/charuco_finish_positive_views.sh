#!/usr/bin/env bash
# Continue only from the measured x6 return; each step validates before action.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
trap 'echo "Calibration stopped; no further view will start" >&2; exit 130' INT TERM
reference=data/commissioning/charuco_reference.npz
current=data/commissioning/board_x06_corrected_view.npz
for spec in x:3:return x:0:return y:3:out y:6:out y:9:out y:12:out y:9:return y:6:return; do
 IFS=: read -r axis angle direction <<< "$spec"
 printf -v step_id 'board_%s%02d_%s2' "$axis" "$angle" "$direction"
 bash tools/charuco_step.sh "$step_id" "$axis" "$angle" "$reference" "$current"
 current="data/commissioning/${step_id}_view.npz"
done
