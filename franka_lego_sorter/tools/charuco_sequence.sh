#!/usr/bin/env bash
# Raised, bounded calibration views; stops on any failed plan/action/observation.
set -euo pipefail
root_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root_dir"
trap 'echo "Calibration sequence interrupted; no following view will start" >&2; exit 130' INT TERM
reference='data/commissioning/charuco_reference.npz'
current='data/commissioning/board_x03_view.npz'
for spec in x:6 x:9 x:12 x:15 x:11 x:7 x:3 x:0 y:3 y:6 y:9 y:12 y:15 y:11 y:7 y:3 y:0; do
 axis="${spec%:*}";angle="${spec#*:}"
 printf -v step_id 'board_%s%02d_seq' "$axis" "$angle"
 echo "Planning measured calibration view $step_id"
 bash tools/charuco_step.sh "$step_id" "$axis" "$angle" "$reference" "$current"
 current="data/commissioning/${step_id}_view.npz"
done
printf 'Calibration sequence complete; final measured snapshot %s\n' "$current"
