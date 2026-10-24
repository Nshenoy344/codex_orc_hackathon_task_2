# 03 — Hand-eye calibration

[Algorithm, rationale, alternatives, observed outcomes and limitations](ALGORITHM_AND_DECISIONS.md).

**Decision outcome:** A 17-view numerical candidate passed checks; physical calibration accuracy remains unverified.

ROS 2 package: [franka_realsense_handeye_ros2](../../sources/franka_realsense_handeye_ros2), actual ROS package name`franka_realsense_handeye`. Its upstream README contains build, launch, capture-service and solver instructions. Parameters include ArUco dictionary, board dimensions, square/marker lengths, frames, topics and acceptance thresholds.

The package consumes color image/CameraInfo and timestamped TF and never commands the arm. It is linked into the copied robot workspace for colcon discovery. The sorter also reuses this package's calibration core through the preserved sibling-directory layout.

For the actual commissioning run, paired direct-SDK RGBD captures and ROS TCP measurements were processed by the sorter's `charuco_handeye.py`. The live board had7x5 squares with4x4-family markers, not the package's example 12x7 board. Actual parameters are saved in `data/commissioning/charuco_center_inspection.json`.

Seventeen measured views were used. The RGBD-plane estimator intersects identified corner rays with an independently fitted depth plane, obtains board pose by rigid fitting, then solves/validates hand-eye geometry. Repeated translation-only views were rejected earlier. Use `--estimator rgbd_plane` explicitly to reproduce that estimator; `tools/solve_charuco_batch.sh` defaults to PnP and is not by itself the command for the accepted RGBD-plane candidate.

Saved candidate: [charuco_rgbd_plane_candidate.yaml](../../sources/franka_lego_sorter/data/commissioning/charuco_rgbd_plane_candidate.yaml). Approximate residuals:0.33 mm trainingRMS,0.85 mm maximum held-out residual,1.71 mm leave-one-out translation spread. Square size29.95 mm was estimated from depth, not ruler-verified. Physical verification remains false.

```text
T_A_B maps column-vector coordinates from B into A.
T_tcp_color: aligned/color optical points -> TCP
T_tcp_depth = T_tcp_color @ T_color_depth
T_depth_tcp = inverse(T_tcp_depth)
p_base = T_base_tcp @ T_tcp_color @ p_color
```

Aligned depth uses color intrinsics/geometry. Native depth uses native depth geometry. Do not interchange their transforms.

Numerical calibration acceptance does not establish safe/accurate physical picking. Default production loading refuses a physically unverified candidate; explicit commissioning tools retain that distinction.
