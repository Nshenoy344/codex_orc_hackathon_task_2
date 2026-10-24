# 01 — Robot control and ROS 2 packages

[Algorithm, rationale, alternatives, observed outcomes and limitations](ALGORITHM_AND_DECISIONS.md).

**Decision outcome:** Measured-state planning and bounded correction enabled isolated trials; geometric review did not predict an internal self-avoidance fault.

Saved source workspace: [sources/robot_ws](../../sources/robot_ws). This is a real copy of the external live workspace's source, including uncommitted modifications; it is not a link to `/home/nshenoy/franka_ros2_ws`.

| Source | Role |
|---|---|
| `src/libfranka` |0.20.4 robot SDK and native protocol |
| `src/franka_description` | Robot/Hand URDF, collision meshes and frame definitions |
| `src/franka_ros2/franka_hardware` | ros2_control hardware adapter, including scheduling restoration |
| `src/franka_ros2/franka_gripper` | Native Hand state, move and grasp actions |
| `src/franka_ros2/franka_fr3_moveit_config` | MoveIt planning, execution and FR3 controller configuration |
| `src/franka_ros2/franka_robot_state_broadcaster` | Robot mode, errors and command-success telemetry |
| `src/franka_ros2/franka_msgs` | Franka-specific message/action definitions |

[Every ROS package has a generated reference README](packages/README.md), alongside its copied upstream source documentation.

## Build in this reference workspace

Use a fresh shell, install ROS/SDK prerequisites recorded in provenance, then:

```bash
cd /home/nshenoy/roboinnovate_hackathon/ros2_stack_reference
source /opt/ros/humble/setup.bash
export PYTHONNOUSERSITE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd sources/robot_ws
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install --executor sequential \
  --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF -DBUILD_TESTS=OFF
source install/setup.bash
```

These are rebuild instructions; the copy operation did not run them. The existing successful workspace was built in Release with testing disabled; the original full build used2workers. Serial building here limits contention. `pinocchio` was resolved from the ROS Humble installation; see the saved libfranka CMake cache.

The physical launch used:

```bash
export ROS_DOMAIN_ID=31
ros2 launch franka_fr3_moveit_config moveit.launch.py \
  robot_ip:=172.16.0.2 use_fake_hardware:=false
```

That command activates real control. Its current commissioning prerequisites are documented in block 06, including the changed fingertips. Do not launch a second controller alongside the live stack.

## Local scheduling correction

`franka_hardware/src/robot.cpp` saves the caller's scheduler before constructing `franka::Robot` and restores it afterward, including exception rollback. This prevents constructor-induced FIFO 99 promotion from leaking into ROS support threads. The controller-manager update loop retains its own FIFO 50 policy. Source, patch artifacts and build evidence are included. Restoring caller scheduling does not change Franka collision thresholds.

The copied source is already patched: do not apply the archived patch a second time.

## Frames

Robot IP172.16.0.2; base`fr3_link0`; TCP`fr3_hand_tcp`; planning group`fr3_arm`; ROS domain31. Standard Hand flange-to-TCP is Rz(-45deg) and Z103.4mm. The obsolete custom174mm TCP correction must not be reused. Replacement tips require their physical contact geometry to be established relative to this frame.
