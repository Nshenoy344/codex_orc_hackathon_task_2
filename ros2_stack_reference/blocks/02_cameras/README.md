# 02 — Cameras and capture

[Algorithm, rationale, alternatives, observed outcomes and limitations](ALGORITHM_AND_DECISIONS.md).

**Decision outcome:** Stationary RGBD worked; High Accuracy removed three inspected false-depth points, with reduced depth coverage.

Main capture implementation: [camera.py](../../sources/franka_lego_sorter/lego_sorter/camera.py). Saved SDK runtime: [pyrealsense2_runtime](../../sources/pyrealsense2_runtime). Its distribution metadata records pyrealsense2 version 2.59.1.11053. It is a platform-specific runtime copy; ROS, NumPy and OpenCV remain system dependencies.

The SDK identified RealSense D435 serial 943222073921. Commissioning used native 1280x720 RGB/depth at 30 fps, depth aligned into color coordinates. Captures contain RGB, depth in metres, intrinsics, actual sensor settings, factory color/depth extrinsics and paired robot TCP pose.

One direct RealSense consumer at a time: RealSense Viewer previously occupied the stream. Auxiliary webcams use separate devices. The optional ROS hand-eye node instead consumes `realsense2_camera` topics; do not start that driver's camera stream alongside direct SDK capture.

High Accuracy depth preset 3 removed three spurious near-depth points in dark gaps in a subsequent actual capture. Presets are configured before streaming and their actual values are recorded. Increased missing depth remains visible; it is not filled to manufacture measurements. Color settings are separate.

The saved auxiliary device roles are in [overview_cameras.yaml](../../sources/franka_lego_sorter/config/overview_cameras.yaml). Device numbering may change; stable `/dev/v4l/by-id/` paths are preferred. Their metric robot extrinsics are unverified, and repositioning invalidates prior view geometry.

Example capture commands from the original application interface:

```bash
bash tools/commission.sh record --config config/inspect_720_high_accuracy.yaml \
  --output data/commissioning/NEW_OBSERVATION.npz
/usr/bin/python3 tools/capture_overview.py --device /dev/video8 \
  --rotation-deg 180 --output data/commissioning/NEW_OVERVIEW.png
```

Commissioning wrappers contain the original live-overlay path; adapt that path when using a rebuilt reference overlay. Capture reads robot state and camera data; it sends no motion.

For new installations, install a matching pyrealsense2 distribution or use the saved runtime only on a compatible architecture/ABI. The copied SDK distribution retains its metadata/license material.
