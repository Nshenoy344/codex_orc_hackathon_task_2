#!/usr/bin/env bash
set -eo pipefail
reference_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONPATH="$reference_root/sources/franka_lego_sorter:$reference_root/sources/franka_realsense_handeye_ros2:${PYTHONPATH:-}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
case "${1:-simulate}" in
  simulate|check) ;;
  *) echo 'This wrapper permits only offline simulate/check; no hardware command.' >&2; exit 2 ;;
esac
if [[ $# == 0 ]]; then set -- simulate; fi
cd "$reference_root/sources/franka_lego_sorter"
exec /usr/bin/python3 -m lego_sorter.cli --config "$reference_root/config/reference_demo.yaml" "$@"
