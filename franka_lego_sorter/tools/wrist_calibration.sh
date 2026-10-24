#!/usr/bin/env bash
set -eo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
source tools/analysis_limits.sh
export PYTHONNOUSERSITE=1
export PYTHONPATH="$PWD:$HOME/.local/share/franka-camera-viewer/python:${PYTHONPATH:-}"
exec /usr/bin/python3 -u tools/wrist_calibration.py "$@"
