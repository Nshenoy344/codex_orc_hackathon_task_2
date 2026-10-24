# ROS 2 LEGO stack reference

Saved 2026-10-24T18:46:21.411483+00:00. This is a separate source-and-evidence copy. The live repositories, installed ROS workspace, devices and running processes were not changed by creating it.

The calibration and sorter were already under `~/roboinnovate_hackathon`; the external robot-workspace sources have now also been saved here. Repository-local edits, including the Franka constructor scheduling restoration, are included. Git object databases and old build/install trees are excluded. Upstream source licenses remain with the copied packages.

## Algorithms and engineering decisions

Read [the algorithm and decision guide](ALGORITHM_AND_DECISIONS.md) for each block's method, rationale, alternatives, observed successes/failures and remaining limitations. Each block README links to its own detailed guide.

## Separate stack blocks

| Block | Contents |
|---|---|
| [00 Environment](blocks/00_environment/README.md) | OS, Python, ROS domain, dependencies and environment setup |
| [01 Robot control](blocks/01_robot_control/README.md) | Actual Franka ROS packages, libfranka, description, MoveIt and local patch |
| [02 Cameras](blocks/02_cameras/README.md) | RealSense capture, auxiliary webcams, settings and SDK runtime |
| [03 Hand-eye calibration](blocks/03_handeye_calibration/README.md) | ROS package, board configuration, transforms and saved candidate |
| [04 Perception](blocks/04_perception/README.md) | Brick footprints, color teaching, template measurements and confidence |
| [05 Sorting and manipulation](blocks/05_sorting_and_manipulation/README.md) | Routing, 12-slot occupancy, alternating 90-degree layers and motion stages |
| [06 Commissioning and recovery](blocks/06_commissioning_and_recovery/README.md) | Physical results, failures, journals, timing fixes and remaining work |

```text
sources/
  robot_ws/src/
    franka_ros2/                 # complete local ROS source tree
    franka_description/          # meshes, URDF and geometry
    libfranka/                   #0.20.4, including populated common submodule
    franka_realsense_handeye_ros2 -> ../../franka_realsense_handeye_ros2
  franka_realsense_handeye_ros2/ # ROS 2 calibration package
  franka_lego_sorter/            # Python application using ROS 2; includes data/ and logs/
  pyrealsense2_runtime/          # saved x86_64 SDK runtime, not a source build
hardware/                       # new fingertip STL and original task screenshot
provenance/                     # source revisions, patches, package versions, build-cache references
config/                         # reference-only application configuration
scripts/                        # environment and offline entry points
```

## Reproduce the software setup

Start with [environment setup](blocks/00_environment/README.md), then [build the robot workspace](blocks/01_robot_control/README.md). Calibration instructions follow in block 03. No build, install, launch or hardware command was performed to create this bundle.

Read-only integrity check and offline routing examples:

```bash
cd /home/nshenoy/roboinnovate_hackathon/ros2_stack_reference
/usr/bin/python3 scripts/verify_snapshot.py
bash scripts/offline_demo.sh simulate
bash scripts/offline_demo.sh simulate --requested 300121 6117418
```

## Physical status at the conversation boundary

One isolated maroon brick was picked and released, but its placement was parallel to the previous layer and off center. It was **not a correct task placement**. Whole-pattern 90-degree alternation is now implemented in software. Input-pallet extraction, correct complete output layers, measured repeated-layer pitch and both full request modes remain unproven.

Replacement fingertips were reported installed, and their STL is preserved in `hardware/`. Their calibrated geometry and full-height grasp have not been established by the evidence described in this handoff. Calibration remains a numerical candidate with physical verification false. Current running main-thread commissioning may progress independently of this point-in-time copy.

Some inherited upstream/application README status paragraphs describe older commissioning stages. Use this overview, block 06 and the dated raw result files together; a controller success message or passing test is not a sorting demonstration.

## Portability and completeness

The two application repositories retain their sibling layout because the sorter loads the existing calibration core by relative path. Original scripts and evidence contain absolute `/home/nshenoy` paths; they are copied faithfully, not silently rewritten. Use the reference wrappers for offline inspection and read the relocation instructions before using inherited launchers on another machine.

`data/` and `logs/` are included even though the original sorter excludes them from Git. ROS binaries and system dependencies are not bundled; rebuild from the saved source and use `provenance/system_packages.tsv` as the installed-version record. The bundled RealSense Python runtime is architecture/Python-ABI dependent.

`source_snapshot_manifest.json` records hashes of copied source/evidence files. `manifest.sha256` covers all regular files in the completed bundle except itself. `symlinks.json` records the small set of internal symbolic links. No link intentionally targets the live workspace.
