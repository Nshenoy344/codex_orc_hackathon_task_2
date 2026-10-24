#!/usr/bin/env bash
# Install and open the official RealSense Viewer, not the Python preview.
set -euo pipefail
log_dir="$HOME/.local/state/franka-camera-viewer"
mkdir -p "$log_dir"
exec > >(tee -a "$log_dir/realsense-viewer.log") 2>&1
if ! command -v realsense-viewer >/dev/null 2>&1; then
    echo 'Installing official RealSense Viewer from the RealSense APT repository.'
    echo 'sudo is required to install system packages and USB device permissions.'
    source /etc/os-release
    if [[ "${ID:-}" != ubuntu || -z "${VERSION_CODENAME:-}" ]]; then
        echo 'This installer requires Ubuntu.' >&2
        exit 1
    fi
    sudo apt-get update
    sudo apt-get install -y ca-certificates curl gnupg
    key_tmp="$(mktemp)"
    trap 'rm -f "$key_tmp"' EXIT
    curl --fail --show-error --location https://librealsense.realsenseai.com/Debian/librealsenseai.asc |
        gpg --batch --yes --dearmor --output "$key_tmp"
    sudo install -d -m 0755 /etc/apt/keyrings
    sudo install -m 0644 "$key_tmp" /etc/apt/keyrings/librealsenseai.gpg
    printf 'deb [signed-by=/etc/apt/keyrings/librealsenseai.gpg] https://librealsense.realsenseai.com/Debian/apt-repo %s main\n' "$VERSION_CODENAME" |
        sudo tee /etc/apt/sources.list.d/librealsense.list
    sudo apt-get update
    sudo apt-get install -y librealsense2-utils librealsense2-udev-rules
    echo 'If the camera is absent, unplug and reconnect its USB cable.'
fi
echo 'Opening official RealSense Viewer. Enable the RGB and Stereo Module streams in the GUI.'
exec realsense-viewer
