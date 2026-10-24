# Franka LEGO sorting and palletizing

ROS 2 Humble implementation for the Franka Research 3 and stock Franka Hand. It reuses the neighboring `franka_realsense_handeye_ros2` calibration core. The task specification is `../Screenshot from 2026-10-24 13-30-06.png`, supplemented by the requested 6×5 input and measured 4×3 output templates.

**Physical sorting is not yet demonstrated.** The operator-confirmed isolated maroon brick was individually grasped and lifted, with source-removal evidence. Its first diagnostic placement was parallel to the reference rather than the required90° first-new-layer pattern, so it is not a successful task placement and has not contributed production occupancy. Both sorting request modes remain unproven.

## Current scene changed: old registrations are ineligible

The operator added support floorboards and may have repositioned the source or output assembly. [Current scene state](data/current_scene_state.json) invalidates **all previous source/template world registrations, support heights, derived placement/recovery targets and old motion plans**. Historical artifacts are retained for review; a previously passing trajectory review does not make those coordinates reusable. No old source pixel association is a current world coordinate.

The12 nonuniform measured reference shapes and their90° wholepattern relationship remain reference geometry only. Obtain fresh paired RGBD and actual TCP, trace/re-register the current12 footprints, and measure the new board/table/support heights before generating replacement targets. The existing offline `inspect_template_observation.py` and `measure_observed_template_faces.py` helpers can reuse the measurement procedure; manual polygons are hash-bound to their actual capture and cannot be copied onto a changed image. Layer pitch remains unmeasured unless independently established from actual settled layers.

At the scene change, the wrist camera was near the old low approach. A close RGB image can document new objects, but it does not establish valid depth or support geometry. A separately reviewed raised observation is needed for fresh RGBD. The calibration candidate remains numerically accepted and physically unverified; moving supports does not itself change the wrist extrinsic. Shade examples retain their physical identity provenance, but ambiguous shades remain UNKNOWN and their historical image/world positions are stale.

## Requested behavior

Relative to the input pallet: requested output on the right, both scrap colors in the tray on the left, and unused good colors in separate upper/lower pallets.

| Color ID | Routing |
| --- | --- |
| `300121`, `6117418`, `6487946` | Good; requested colors go right, others upper/lower |
| `6135191`, `4211201` | Scrap tray only when confidently identified |

Configure `job.requested_colors` as one or two good IDs. `requested_count` is total requested-pallet quantity, not a per-color quota; mixtures accept requested colors in arrival order. The journal reports actual composition. A quota stops the whole run when reached. `source_count` can additionally bound total processing. Unknown shades are reobserved and then cause a recoverable stop, never automatic rejection.

Each good destination has **12 independently measured positions: four parallel ridges of three bricks**. Fill all 12 before the next layer. Pallet spacing is taken from the physical template, including gaps and staggering; no solid rectangular grid or single tower is substituted. Every output has independent occupancy/layer counters.

## Hardware and measured evidence

- Actual SDK camera identification: **RealSense D435, serial `943222073921`**, not D435i. Depth is aligned to color. Use `T_tcp_color` with these pixels.
- Logitech C920 overview was identified at `/dev/video8`; rerun `v4l2-ctl --list-devices` if device numbering changes.
- Robot IP `172.16.0.2`, ROS base `fr3_link0`, TCP `fr3_hand_tcp`, group `fr3_arm`.
- Deployment ROS domain **31**. Launch scripts deliberately use `LEGO_ROS_DOMAIN_ID` (default 31), because this machine also has an unrelated inherited `ROS_DOMAIN_ID=23`.
- Desk now selects standard **Hand**: flange→TCP is `Rz(-45°)` and 103.4 mm Z, matching stock ROS FK. The historical custom Desk TCP at 174 mm required a 70.6 mm correction; **do not apply that obsolete correction to current standard-Hand data**.
- Current standard-Hand model is 0.73 kg with zero additional load. That does not measure the wrist camera/bracket inertia. Do not invent payload values from this README.
- A prior +20 mm retreat measured +19.64 mm Z and about 0.80 mm lateral tracking error. The old controller later stopped on a Cartesian reflex during hold; its cause was not established.
- After Hand reconnection, a further slow upward step reached a settled pose but failed strict endpoint verification at about **2.3 mm** error. Further automatic steps stopped; measured capture is `data/commissioning/resumed_lift_01_actual.npz`. The controller reported trajectory success, illustrating why separate measured-pose verification matters.

Captures, trajectory plans, independent FK reviews, hardware probes, and outcomes are under `data/commissioning/`. These files are local runtime evidence, excluded from Git. Three earlier near-translation-only captures **do not establish a hand-eye transform**; `scene_handeye_rejected.yaml` records the rejection. The existing camera model, templates, and station geometry must not be assumed commissioned because their tools exist.

## Software paths

- `camera.py`: stationary median RGB/depth bursts, aligned color geometry, camera identity/settings metadata.
- `perception.py`: physical shade prototypes in Lab with distance/margin rejection.
- `pallet.py`: multiple observed 2×4 faces, robust local planes, both 16 mm and 32 mm closing directions, measured jaw-footprint clearance. Split or clipped faces remain non-executable.
- `template_registration.py`: validates 12 measured support faces/slot polygons and groups actual 4×3 ridges while retaining measured positions. Template type (`brick_layer`, `fixture`, `base`) and support height are explicit. It does not infer vertical layer pitch.
- `occupancy.py`: measured base-frame TCP placement registry, 12 slots per destination, measured pitch, immutable geometry signature.
- `planner.py`: routing, composition, occupancy, atomic pending-transfer journal and explicit recovery.
- `pallet_runner.py`: actual pallet candidate selection, dynamic grasp width/yaw, selected-source-footprint removal checks, measured-slot placement, and footprint color/depth verification.
- `motion.py`: MoveIt/Franka actions, fresh-state checks, complete Cartesian paths, separate plan/review/execute, measured endpoint checks. No auto-homing on construction.
- `scene_registration.py` / `scene_handeye.py`: optional static-pallet RGBD calibration with rigid registration, observability and held-out checks. An accepted numerical candidate still needs physical validation.
- `commissioning.py` / `calibration_motion.py`: explicit, bounded commissioning plans and stationary observations. Rotation plans have independent dense FK/joint-limit validation and hash-pinned reports.

`runner.py` is the earlier single-stack implementation retained for regression tests. The physical CLI uses **PalletRunner**, not that legacy tower runner.

## Launch and logs

From this directory, use the system Python/ROS stack rather than user-site NumPy 2, which conflicts with the installed OpenCV ABI:

```bash
cd /home/nshenoy/roboinnovate_hackathon/franka_lego_sorter
bash tools/demo.sh check
bash tools/demo.sh simulate
bash tools/demo.sh simulate --requested 300121 6117418
```

Readiness is reported separately from routing simulation:

```bash
bash tools/start_robot.sh
# In another terminal:
bash tools/demo.sh robot-status
```

`start_robot.sh` starts the real installed MoveIt/Franka stack; it holds the measured pose and sends no application trajectory or grasp goal. Do not run it alongside another arm controller. It can activate a torque controller, so inspect fresh mode/errors and the physical workspace before startup. `start_hand.sh` is for a standalone Hand state connection; stop that specific process before a full stack launch so two clients do not compete.

```bash
tail -f logs/demo.log
tail -f logs/commissioning.log
```

## Camera and calibration commands

Close RealSense Viewer and other direct camera consumers before capture:

```bash
bash tools/demo.sh camera preview
bash tools/demo.sh camera snapshot --output data/source.npz
bash tools/commission.sh record --output data/commissioning/observation.npz
PYTHONNOUSERSITE=1 /usr/bin/python3 tools/capture_overview.py --output data/overview.png
```

The commissioning record includes measured `T_base_tcp` before/after capture and rejects movement during acquisition. Hardware commands need normal host USB/network access; restricted sandboxes may fail even when the device itself works.

For ChArUco calibration, follow `../franka_realsense_handeye_ros2/README.md` using the same ROS TCP/base frames. Alternatively, the markerless scene path requires actual multi-axis wrist rotations and a stationary scene, not merely repeated images or translations:

```bash
PYTHONNOUSERSITE=1 /usr/bin/python3 -m lego_sorter.scene_handeye --help
PYTHONNOUSERSITE=1 /usr/bin/python3 -m lego_sorter.calibration_motion --help
```

The live board was identified as 7×5 squares with 4×4 ArUco markers. Its square scale is an aligned-depth estimate (about 29.95 mm), not a ruler-verified print dimension. Actual measured robot poses are paired with stationary RGBD captures. `tools/charuco_step.sh` plans, reviews, executes and records one bounded raised view, stopping on failed checks. `tools/solve_charuco_batch.sh` uses an explicit capture list and writes a hash manifest; missing views prevent a solve.

The acquisition script explicitly uses a uniform 0.8 px reprojection cap for candidate observations. At the observed 0.37 m range and approximately 610 px focal length, this corresponds to about 0.49 mm image-plane residual; it does not bound 3D pose error. Depth-plane consistency, multi-axis excitation, robot/board motion consistency, held-out scene residuals and solution stability remain separate acceptance checks. A numerically accepted candidate still requires physical validation before picking. Rejected captures and failed motion checks remain in `data/commissioning/` and `logs/`.

Standard sorting requires `physical_commissioning_verified: true` in the calibration artifact; a numerically accepted candidate with that field false is refused. `bash tools/demo.sh inspect-calibration-candidate` is read-only. Commissioning point-trial code can explicitly call `load_handeye_candidate`; the full sorting command has no unverified-candidate override. Numerical acceptance must not be promoted to physical verification without saved physical test evidence.

Calibration rotation plans are restricted to a raised, measured TCP region and small reviewed steps with fresh RGBD overlap/depth checks. Numerical FK checks cover the supplied robot model; the wrist camera/bracket is not yet represented in that collision model.

## Physical shade and geometry records

At the measured observation pose, choose an input-only ROI and teach actual known examples under consistent camera settings:

```bash
bash tools/demo.sh camera select-roi
bash tools/demo.sh camera teach --label 300121
bash tools/demo.sh camera teach --label 6117418
bash tools/demo.sh camera teach --label 6487946
bash tools/demo.sh camera teach --label 6135191
bash tools/demo.sh camera teach --label 4211201
```

Select brick interior patches, avoiding background and specular highlights; repeat examples at actual expected views. No physical model has been substituted with screenshot RGB guesses. Camera settings/identity are part of the model; re-teach after changing them. Source perception supports two jaw orientations but requires measured clearance for both opposed fingers. The operator measured a 13 mm finger, 14.5 mm interbrick gap, and only 2 mm of usable tip before the wider screw. Configuration uses 1.5 mm insertion and a 2 mm maximum, with a 1 mm total opening margin. The modelled tip extends 9.5 mm beyond TCP (visual CAD about 8.9 mm); the runner computes TCP from the transformed tip and checks actual toe-corner depths. Before descent, measured jaw opening must match the narrow command within 0.1 mm and preserve the measured gap. The old below-surfaceTCP formula is rejected by the pallet runner. The dense layered pallet may contain attached bricks; this must be determined by physical separation tests, not by a camera assumption.

Shallow interbrick grasping requires a **measured body-top reference**, not the tops of the studs. `pallet.require_body_top_reference: true` fits the central inter-stud strip and requires independently observed raised-stud relief; unresolved depth stops candidate execution. The code never subtracts a nominal stud height to invent a grasp surface. Plane RMS does not establish absolute depth or hand-eye accuracy. For a fresh native 1280×720 capture, set both camera dimensions in the capture configuration; the SDK applies that profile to native color and depth before alignment. Inspect the saved NPZ offline with:

```bash
PYTHONNOUSERSITE=1 /usr/bin/python3 tools/inspect_body_top.py data/commissioning/source_1280.npz --config config/demo.yaml --require-one-verified
```

This saves measured body/stud diagnostics and overlays under `data/commissioning/body_top_inspection/`; exit status 2 means no body reference passed. Support counts scale with physical sampled area and pixel resolution. Passing this inspection alone does not authorize a grasp or prove calibration.

Record `observation`, `pick_reference`, and `scrap` with `teach-pose`. These commands read TF only. Output placement positions instead come from the **measured template registry**, not one taught destination pose:

```bash
bash tools/demo.sh teach-pose observation
bash tools/demo.sh teach-pose pick_reference
bash tools/demo.sh teach-pose scrap
```

`templates_file` uses `format: measured_pallet_templates_v1`, explicit base/TCP frames, and `destinations: requested/upper/lower`. Each required destination supplies:

- `kind: base`, `fixture`, or `existing_brick_layer` after inspecting the actual support.
- `first_place_reference: tcp_pose_for_first_new_layer`.
- Exactly 12 measured `first_place_poses` (4×4 TCP matrices), in ridge-major fill order. Their height must already account for support/template type and actual grasp/release convention.
- Independently **measured** `layer_pitch_m`. Do not assume loose and snapped LEGO have the same pitch.

Output layers interlock by rotating the **whole12-position pattern90°** about the measured template center on every new layer: both slot XY positions and TCP orientations rotate. Each template requires explicit `rotation_center_base_m`, `layer_rotation_deg:90` and `first_new_layer_rotation_from_reference_deg:90`. `first_place_poses` always mean the **first NEW layer**, already rotated90° relative to the existing reference layer. Generate these separately with `first_new_layer_from_reference_aligned_poses(poses, measured_center)` after establishing their actual first-new-layer TCP height; the helper preserves height and supplies no nominal support offset. The registry leaves these first poses unchanged, then rotates later layers by `k*90°` and adds `k*layer_pitch_m` in base Z. The full nonuniform measured pattern is retained; bricks are not merely turned in place. A90° pattern may extend beyond the original footprint, so measure support coverage and review the rotated collision envelope before physical execution. Rotation alone does not prove stud engagement or stable interlocking. Template geometry is hashed into the journal so moved/replaced templates cannot silently reuse occupancy. `execution.max_pallet_layers` limits height; `template_grasp_orientation` records the grasp convention used to measure placement poses. Across-length picks are compensated so brick orientation at the template remains consistent.

[Actual-camera review of the desired90° first-new-layer pattern](data/commissioning/rotated_first_layer_camera_review.png): cyan shows all12 transformed measured footprints, green marks proposed first slot `r1b2`, and red shows the independently observed wrongly oriented diagnostic brick. This is a geometry review, not verified support or successful stacking.


## Execute and recover

### Measured commissioning result (24 October 2026)

One isolated, operator-confirmed **6117418 maroon** brick was grasped with the stock Hand at15N. The measured jaw width remained about15.35mm through a9.75mm proof lift and a subsequent raise to TCP Z≈300mm. A separate RGBD observation found the original footprint empty:99.9% valid depth,100% within3mm of the measured floor, and median absolute floor residual0.54mm. Evidence is in `data/commissioning/isolated_first_grasp.json`, `isolated_first_lift10_result.json`, `isolated_carry_observe300_checkpoint_result.json`, and `isolated_first_pick_source_proof.json`.

This establishes one isolated pick. Input-pallet extraction, placement, layer progression and complete request-mode demonstrations have **not** yet been established. The brick remains held until a reviewed placement. The grasp used measured floor support with explicitly nominal brick dimensions; it does not certify a measured body-top reference or the tight pallet grasp. Calibration remains marked physically unverified.

The subsequent template-view trajectory completed in the ROS controller, but endpoint verification failed at2.38mm error (2mm limit). Its failed result is retained; geometry uses the actual paired TCP instead of the commanded target. After the operator removed an occluding loose brick, all12 original orange template footprints passed strict4×3 grouping (`template_reference_uncovered_interior_registration.json`). This establishes observed geometry, not layer pitch or stud engagement.

Physical execution requires a valid hand-eye result, all five physical shade references, registered templates, measured observation/pick/scrap poses and workspace/transit bounds. The default 13-brick request is intended to exercise a full layer plus the first slot above it; it is not evidence that either was achieved.

```bash
bash tools/demo.sh run --execute --max-transfers 1
bash tools/demo.sh run --execute
```

Each transfer writes a pending reservation before actuator commands. Counters, per-shade composition and 12-slot occupancy advance only after verified release/placement. Unknown colors, incomplete plans, tracking/grasp failures, and source/destination verification failures stop recoverably. Physical multi-brick attachment checks remain a commissioning limitation: current source vision verifies after the initial lift, not a proven force-controlled de-studding routine.

```bash
bash tools/demo.sh pending
bash tools/demo.sh reconcile TRANSFER_ID placed --physical-state-verified
# Or, only if pre-transfer source/destination state has physically been restored:
bash tools/demo.sh reconcile TRANSFER_ID returned --physical-state-verified
```

Reconciliation records an inspected physical outcome; it never moves hardware. Do not clear pending journals just to force retries. Do not reconcile while another sorter process is active. Failed commissioning trajectory verification may mean the arm already moved; inspect its actual pose and saved plan instead of repeating a displacement.

## Verification

```bash
export PYTHONNOUSERSITE=1
source /opt/ros/humble/setup.bash
source /home/nshenoy/franka_ros2_ws/install/setup.bash
/usr/bin/python3 -m unittest discover -s test -v
/usr/bin/python3 -m unittest discover -s tests -v
```

Tests include measured/nonuniform 12-slot geometry, layer progression, both request modes/composition, ambiguous-color stops, pending recovery, source and destination RGBD checks, dynamic grasp orientation, pose conventions, registration observability, and rejected calibration data. Hardware success rate, brick attachment/separation, layer pitch, output geometry, physical shade confusion, and both complete sorting demonstrations remain to be measured.


## Analysis CPU limits

Local launchers cap OpenBLAS/OMP/MKL/NumExpr and OpenCV/TBB analysis to one thread by default (`LEGO_ANALYSIS_THREADS=1`; explicit values1–4 are supported). This is a reversible child-process setting, not a robot/kernel scheduling change. An idle `communication_constraints_violation` occurred281.75seconds after a completed move; CPU contention is a plausible cause, not a proven diagnosis. The controller had successfully obtained FIFO priority50, Ethernet was1Gb/s full-duplex, and link error/drop counters were zero. Keep these limits during hardware commissioning.

On the inspected host, eno1 IRQ activity was concentrated on CPU8 and camera IRQ activity on CPU10. Optionally run expensive **offline analysis only** on other cores, e.g. `taskset -c 0,1,4-7 bash tools/demo.sh simulate`; do not apply that example blindly to the realtime controller or change system IRQ affinity. Inspect live controller scheduling after each restart.


## Observed wrist self-avoidance fault and recovery

The15-second `isolated_hover200_joint_high_plan.json` noncontact observation move aborted after14.508seconds with Franka `self_collision_avoidance_violation` (`bringup_after_comm_fault.log`, line243). The later60-second ROS timeout was a consequence of the controller process terminating. This was a failed physical move, not a successful pick or demonstration. The previously passing review checked URDF joint limits and observed environmental geometry; it did not reproduce the robot's internal self-avoidance model. The recorded reflex configuration had joint5≈2.511rad and joint6≈3.809rad. Exact contact or the proprietary internal avoidance boundary has not been established.

Do not replay that plan; its `.revoked.json` records the failure. The shared Cartesian planner/executor now rejects trajectories exceeding the application wrist branch bounds **−1.5≤joint5≤1.5rad, joint6≤3.0rad**, including sampled spline excursions between waypoints. The isolated joint-space planner also sends these as path constraints. These are conservative application restrictions after the observed fault, not changes to robot collision thresholds and not a guarantee of self-clearance.

Recovery requires the operator to clear the reflex in Desk and hand-guide the wrist into an open configuration within the application bounds. Do not send an automatic recovery or replay an old trajectory. After that physical action, read fresh robot state, restart the existing stack only if normal control is available, recapture the moved scene, and generate/review a new plan from the measured configuration. Reobserve the isolated brick before any contact stage; the previous hover target was an observation reference and no longer establishes its actual grasp location.

## Temporary host latency guard and cleanup order

Repeated idle `communication_constraints_violation` faults continued with analysis thread limits. Host inspection found all CPU governors at `schedutil`, an enabled C3 idle state reporting350µs latency, and active Ethernet EEE. The temporary administrator guard now holds performance governors, a zero-microsecond `/dev/cpu_dma_latency` request, and EEE disabled on `eno1`. These settings address plausible latency sources; stable control must still be measured. The guard opens no robot connection.

The guard alone did **not** eliminate faults: another occurred about34seconds after the template-view move, followed by an idle fault after restarting. Source/process inspection found the libfranka Robot constructor promoting its calling thread to FIFO99; ROS support threads inherited99 while the explicit controller update thread ran FIFO50. `tools/normalize_controller_priorities.py` can audit a specific live controller PID and, with `--apply`, demote only its FIFO99 threads to ordinary scheduling while preserving the FIFO50 update thread. It refuses unexpected process identities or missing/ambiguous update threads and writes an audit record. A source-level constructor scheduling restoration is being prepared; no improved reliability is claimed before measurement.

```bash
pkexec /usr/bin/python3 /home/nshenoy/roboinnovate_hackathon/franka_lego_sorter/tools/robot_host_performance.py \
  --state-file /home/nshenoy/roboinnovate_hackathon/franka_lego_sorter/data/commissioning/host_performance_active.json \
  --interface eno1 --disable-eee
```

Keep that process running during robot control. Inspect its state JSON for `active:true`, every governor at `performance`, and `current_eee.eee_enabled:0`. The read-only ROS timing monitor uses a depth-one subscription and processes at most100callbacks/s; its message gaps describe this subscriber, not direct Ethernet/FCI packet latency. Its `control_command_success_rate` values come from the robot-state message.

Cleanup order matters:

1. Finish/stop arm commands, then stop the ROS controller/robot stack in its launch terminal. Confirm the controller is no longer running.
2. Stop the latency guard with **Ctrl+C in its administrator terminal**, or send SIGINT/SIGTERM to its recorded PID using administrator privileges. Restoring EEE can renegotiate the Ethernet link, so never stop this guard while the controller is active.
3. Inspect the state JSON: `active:false`, no `restore_errors`, original CPU governors restored and original EEE configuration restored. Closing the guard's latency FD releases the PM QoS request.

Do not use SIGKILL for routine cleanup: it closes the latency FD but cannot run governor/EEE restoration. If the guard is killed or the host fails, its saved `original_governors` and `original_eee` remain the restoration record; inspect actual settings before restarting control. This guard does not change or restore the separately selected desktop power profile.

The separate isolated-contact camera envelope uses conservative color-optical-frame bounds X±100mm, Y±25mm, Z−40..+20mm, plus2mm margin. It encloses the vendor D435 URDF collision box transformed using the capture's actual SDK `T_color_depth`; it is not a measured camera-mount CAD model. Source: [RealSense D435 URDF](https://github.com/realsenseai/realsense-ros/blob/ros2-master/realsense2_description/urdf/_d435.urdf.xacro). The nominal vendor depth-optical box dimensions are90×25×25.05mm; its transformed front extends beyond+10mm, which is why the separate envelope now allows+20mm.

The close observation placed the floor around259mm optical depth and produced unreliable brick-face depth despite reliable local floor-plane fits. The [D435 specifications](https://www.realsenseai.com/products/stereo-depth-camera-d435/) list approximately280mm minimum depth at1280 resolution. Future brick-depth observations should use at least320mm optical distance. The isolated trial therefore explicitly distinguishes measured floor support and assumed nominal LEGO geometry from a measured brick-body surface; it does not lower depth-validity thresholds to force a detection.

## Constructor scheduler restoration in the installed Franka hardware adapter

Further communication faults occurred even with the CPU/NIC guard active. Read-only inspection found the controller-manager update thread at FIFO50, but the main thread and many ROS/DDS descendants at FIFO99. The launch ancestry was ordinary scheduling. Local libfranka0.20.4 explicitly promotes its constructor's calling thread to FIFO99; the synchronous hardware adapter did not restore that caller afterwards, allowing later descendants to inherit priority99 above the actual update loop.

The narrow patch in `data/commissioning/franka_constructor_scheduler_restore.patch` saves the incoming thread policy/priority around **only** `franka::Robot` construction and restores it immediately afterwards. RAII also restores on exceptions. Libfranka's enforced realtime/kernel checks and the controller-manager's explicit realtime update policy remain unchanged. The installed `franka_hardware` library was rebuilt alone with one worker and analysis thread caps; the build log and an offline fake-scheduler C++ regression are saved alongside the patch. That regression checks normal restoration, constructor-exception rollback, explicit restore failure, preservation of an incoming FIFO50 policy and failed scheduler capture, without opening a robot connection or changing host scheduling.

The patch was built and installed successfully (only `franka_hardware`, one worker); the exact patched RAII class passed its offline normal/exception/failure rollback tests. Runtime inspection after restart confirmed controller PID141677 had its main and ancillary threads at SCHED_OTHER, exactly one update thread TID141798 at FIFO50, and no FIFO99 descendants. The subsequent first loose-placement approach physically succeeded with holding width unchanged. This verifies the priority-leak correction and that maneuver; it does not establish that all communication faults are solved. Keep measuring command success/faults, and recheck thread policies after future restarts. The CPU/NIC guard cleanup order still applies.
