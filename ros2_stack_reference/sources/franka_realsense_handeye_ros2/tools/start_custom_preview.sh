#!/usr/bin/env bash
set -euo pipefail
viewer_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
sdk_dir="$HOME/.local/share/franka-camera-viewer/python"
log_dir="$HOME/.local/state/franka-camera-viewer"
mkdir -p "$sdk_dir" "$log_dir"
exec > >(tee -a "$log_dir/viewer.log") 2>&1
export PYTHONNOUSERSITE=1
export PYTHONPATH="$sdk_dir"
if ! /usr/bin/python3 -c 'import pyrealsense2' 2>/dev/null; then
    /usr/bin/python3 -m pip install --target "$sdk_dir" pyrealsense2
fi
/usr/bin/python3 -c 'import cv2, numpy, pyrealsense2; print("RealSense, OpenCV, NumPy imports OK", flush=True)'
exec /usr/bin/python3 -u "$viewer_dir/view_camera.py"
