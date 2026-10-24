# 00 — Environment

[Algorithm, rationale, alternatives, observed outcomes and limitations](ALGORITHM_AND_DECISIONS.md).

**Decision outcome:** Compatible imports and ROS discovery worked; thread limits alone did not fix communication faults.

Reference host: Ubuntu 22.04.5, ROS 2 Humble, Python 3.10, x86_64, PREEMPT_RT kernel 5.15.0-1116-realtime. Exact installed dependency versions are in [system_packages.tsv](../../provenance/system_packages.tsv); build cache files record original dependency locations, not reusable CMake configurations.

Use `/usr/bin/python3` and `PYTHONNOUSERSITE=1`: user-site NumPy 2 conflicted with system OpenCV/cv_bridge. The application declares NumPy<2. Preserve the ROS-generated PYTHONPATH when prepending local modules.

```bash
source /opt/ros/humble/setup.bash
export PYTHONNOUSERSITE=1
export ROS_DOMAIN_ID=31
```

The recorded live overlay was `/home/nshenoy/franka_ros2_ws/install/setup.bash`; a rebuilt reference overlay is `sources/robot_ws/install/setup.bash`. Do not source ROS setup under `set -u`; its setup scripts may reference unset variables.

Analysis launchers cap OpenBLAS, OMP, MKL, NumExpr and OpenCV threads to 1. These are child-process settings, separate from the administrator latency guard.

`scripts/reference_env.sh` provides the copied module paths and sources a rebuilt reference overlay if present. It does not launch ROS nodes, change scheduler settings or open hardware.

Dependency installation examples for a matching Ubuntu/ROS environment are in the copied calibration and robot READMEs. This archive preserves versions but does not claim a clean-machine dependency installation was tested.
