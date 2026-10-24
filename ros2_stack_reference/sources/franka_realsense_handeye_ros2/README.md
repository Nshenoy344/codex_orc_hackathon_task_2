# Franka–RealSense ROS 2 hand-eye calibration

ROS 2 Python package for an **eye-in-hand** RealSense rigidly attached to a Franka gripper, observing a **stationary ChArUco board**. It replaces the neighboring `realsense_callibration-main` routine's direct Rokae commands and RealSense SDK capture with ROS image topics, CameraInfo, and timestamped TF. It never commands the robot. Move the Franka using your established Franka/MoveIt or hand-guiding workflow, stop, then capture.

## Result and transform direction

`T_A_B` maps column-vector coordinates in B into A. Translation units are meters.

The YAML result contains:

- `T_tcp_color`: color optical coordinates → gripper TCP coordinates.
- `T_tcp_depth`: native depth optical coordinates → gripper TCP coordinates.
- `T_depth_tcp`: gripper TCP coordinates → native depth optical coordinates, the inverse of `T_tcp_depth`.
- `T_color_depth`: factory depth optical → color optical extrinsic from RealSense TF.
- Sample poses, intrinsics, detection errors, parameters, and board consistency metrics.

For a native depth point `p_depth`, use `p_tcp = T_tcp_depth @ [x, y, z, 1]`. For TCP coordinates in the camera depth system use `T_depth_tcp`. To place depth points in the robot base, use `p_base = T_base_tcp @ T_tcp_depth @ p_depth` at the depth acquisition time.

The solver uses `T_base_tcp` and `T_color_board`, returning `T_tcp_color`. We compose `T_tcp_depth = T_tcp_color @ T_color_depth`. Calibration uses color imagery because the printed board has reliable intensity corners; depth values are not needed for this extrinsic solve. Optical axes are x right, y down, z forward. Depth **aligned to color** uses the color optical geometry: use `T_tcp_color` for points deprojected with aligned-to-color intrinsics. Do not substitute native depth intrinsics for aligned images. Depth scale/intrinsic calibration is outside this package.

## Requirements and build (ROS 2 Humble)

Requires an existing working `franka_ros2` setup publishing base-to-TCP TF, and `realsense2_camera` publishing color image/CameraInfo and internal color/depth TF. Franka package launch commands vary by installed robot/driver version; launch the robot with your existing configuration. No robot IP or controller parameters are necessary here because the calibration node only reads TF.

```bash
source /opt/ros/humble/setup.bash
sudo apt install python3-colcon-common-extensions python3-rosdep python3-opencv python3-numpy python3-yaml ros-humble-cv-bridge ros-humble-tf2-ros
mkdir -p ~/franka_calibration_ws/src
ln -s /home/nshenoy/roboinnovate_hackathon/franka_realsense_handeye_ros2 ~/franka_calibration_ws/src/
cd ~/franka_calibration_ws
rosdep install --from-paths src --ignore-src -r -y
PYTHONNOUSERSITE=1 colcon build --symlink-install
source install/setup.bash
```

Use `/usr/bin/python3` and a consistent system OpenCV/NumPy/cv_bridge installation. On this machine, user-site NumPy 2 conflicts with system OpenCV; `PYTHONNOUSERSITE=1` avoids that conflict without changing installed packages. Do not install a second pip OpenCV over the ROS environment. The code supports OpenCV 4.5's legacy board API and the newer CharucoDetector API. Board patterns changed in OpenCV 4.6, especially for even row counts; check the actual printed pattern and `legacy_pattern` when changing OpenCV versions. The supplied seven-row board matches the original routine's dimensions and dictionary.

## Configure

Copy `config/calibration.yaml` somewhere writable, edit it, and pass its absolute path to the launch file. Parameters are read at startup; restart after edits. Runtime parameter changes are rejected to avoid mixing calibration settings.

| Parameter | Purpose/default |
| --- | --- |
| `base_frame` | Franka fixed base, `fr3_link0` |
| `tcp_frame` | Actual gripper TCP, `fr3_hand_tcp`; verify against your URDF |
| `color_frame`, `depth_frame` | `camera_color_optical_frame`, `camera_depth_optical_frame` |
| `image_topic`, `camera_info_topic` | Raw color image and its CameraInfo under `/camera/camera/color/` |
| `dictionary` | OpenCV symbolic ArUco type, e.g. `DICT_5X5_1000`, `DICT_4X4_50`, `DICT_6X6_250` |
| `squares_x`, `squares_y` | Number of squares (not inner corners), 12 × 7 |
| `square_length_m`, `marker_length_m` | Measured printed dimensions, 0.040 and 0.030 m |
| `legacy_pattern` | Use old pattern layout on newer OpenCV; default false |
| `min_charuco_corners` | Minimum accepted corners, 20 |
| `max_reprojection_error_px` | Maximum per-image RMS, 0.9 px |
| `min_samples` | Minimum paired captures, 12; aim for 20–30 |
| `method` | `PARK`, `TSAI`, `HORAUD`, `ANDREFF`, or `DANIILIDIS` |
| `max_image_age_sec` | Reject stale/future images, 1.0 s |
| `tf_timeout_sec` | Wait for exact image-time TF, 0.2 s |
| `min_pose_translation_m`, `min_pose_rotation_deg` | Reject poses close to any prior sample in both measures, 0.015 m / 5° |
| `min_rotation_span_deg` | Lower bound on second singular value of relative rotation vectors, 15°; rejects parallel-axis/low-excitation sets |
| `max_board_translation_rms_m`, `max_board_rotation_rms_deg` | Reject solutions with moving reconstructed board, 0.01 m / 2° |
| `output_file` | Result YAML; paired captures also saved as `<output_file>.samples.yaml` |
| `use_sim_time` | Set true only with a matching `/clock` source |

Franka frame prefixes vary: Panda may use `panda_link0`/`panda_hand_tcp`; custom tools may define a different TCP. Choosing the flange yields a flange calibration, not a TCP calibration. Define the actual rigid flange-to-tool transform in the robot description first. Camera topic namespaces and frame names may also differ. Inspect them instead of assuming:

```bash
ros2 topic list
ros2 topic echo /camera/camera/color/camera_info --once
ros2 run tf2_ros tf2_echo fr3_link0 fr3_hand_tcp
ros2 run tf2_ros tf2_echo camera_color_optical_frame camera_depth_optical_frame
```

Only the robot base→TCP and internal camera color→depth transforms are required. A preexisting robot→camera transform is not needed. All publishers must use a consistent time base. CameraInfo must describe the raw color stream with a supported pinhole distortion model (`plumb_bob` or `rational_polynomial`).

## Board and capture workflow

1. Use your existing board if it matches the YAML exactly. Otherwise generate one:

   ```bash
   PYTHONNOUSERSITE=1 ros2 run franka_realsense_handeye generate_board --params /absolute/path/calibration.yaml --output /tmp/charuco.png
   ```

   The default board area is **480 × 280 mm**. Print at physical size without scaling, add a white outer margin, mount flat, and measure the actual square size. Set YAML dimensions from the printed result. This is a ChArUco board, not a single ArUco marker.

2. Fix the board to the workspace and camera to the tool. Launch your Franka TF stack. Launch RealSense separately, for example (adjust arguments for your installed wrapper):

   ```bash
   ros2 launch realsense2_camera rs_launch.py enable_color:=true enable_depth:=true publish_tf:=true
   ```

3. Start calibration:

   ```bash
   PYTHONNOUSERSITE=1 ros2 launch franka_realsense_handeye calibration.launch.py params_file:=/absolute/path/calibration.yaml
   ```

4. Move to distinct stationary poses with the board in view, allow vibration/exposure to settle, then capture each pose:

   ```bash
   ros2 service call /handeye_calibrator/capture_sample std_srvs/srv/Trigger '{}'
   ros2 service call /handeye_calibrator/status std_srvs/srv/Trigger '{}'
   ```

   Vary translation, distance, and tilt about at least two axes while preserving visibility. Pure translations or rotations about one axis cannot determine a reliable full transform. Capture refuses stale images, repeated frames, nearby poses, missing exact-time TF, changed intrinsics, too few corners, and excessive reprojection error. The operator is responsible for ensuring the robot is stationary; motion settling is not detected automatically.

5. Solve after at least 12 accepted samples:

   ```bash
   ros2 service call /handeye_calibrator/solve std_srvs/srv/Trigger '{}'
   cat ~/franka_handeye_results/calibration.yaml
   ```

   A successful service saves both requested transform directions. Board consistency thresholds reject poor fits; these are training residuals, not an independent accuracy guarantee. Verify on new held-out robot poses and a known physical point before using the result. The node does not automatically discard outliers.

6. To start a new capture session:

   ```bash
   ros2 service call /handeye_calibrator/reset std_srvs/srv/Trigger '{}'
   ```

   Captures persist for inspection but are not automatically reloaded. Reset overwrites the sample sidecar and retains the last solved result. Back up files or choose a new output path for separate sessions.

## Using the result in TF

The result is saved rather than automatically broadcast. RealSense already parents the optical frames under `camera_link`. To integrate without giving an optical frame two parents, obtain `T_color_camera_link` from the camera's internal TF and compute `T_tcp_camera_link = T_tcp_color @ T_color_camera_link`. Add that single fixed joint from the actual TCP to `camera_link` in your robot description or static transform publisher. Preserve the wrapper's existing internal TF tree. Ensure any old mount transform is replaced rather than duplicated.

## Verification

```bash
cd /home/nshenoy/roboinnovate_hackathon/franka_realsense_handeye_ros2
PYTHONNOUSERSITE=1 /usr/bin/python3 -m unittest discover -s test -v
```

Synthetic tests cover all five solvers, a nonidentity depth extrinsic and inverse point mapping, insufficient/degenerate poses, and board image detection with pose recovery. The package also builds with colcon on ROS 2 Humble and the node reached its ready state. Live DDS communication could not be verified in the development sandbox because network sockets are restricted. Hardware calibration still requires the real robot, camera, and printed board. This is a new implementation informed by the neighboring routine; the original files are unchanged. No upstream redistribution license was supplied, so this repository does not assert an open-source license.

References: [OpenCV hand-eye conventions](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html), [ChArUco detection and pattern compatibility](https://docs.opencv.org/4.x/df/d4a/tutorial_charuco_detection.html), [RealSense ROS wrapper](https://github.com/realsenseai/realsense-ros), [Franka ROS 2 TCP configuration](https://frankarobotics.github.io/docs/doc/franka_ros2_jazzy/franka_fr3_moveit_config/doc/index.html).
