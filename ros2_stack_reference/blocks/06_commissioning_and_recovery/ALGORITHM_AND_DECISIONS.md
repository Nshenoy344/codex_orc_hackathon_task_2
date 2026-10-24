# Evidence-driven commissioning and failure handling — algorithm and decisions

## Method and algorithm

Commission one observable transition at a time. Before acting, save inputs, intended pose/width, plan and review. Afterward, distinguish controller result, measured robot state, gripper feedback, source change and destination observation. Keep an uncertain transfer pending; reconcile only after inspecting its actual physical outcome.

The evidence layers answer different questions:

| Evidence | What it establishes | What it does not establish |
|---|---|---|
| Action result | Controller/API completion | Object picked or correctly placed |
| Measured TCP | Tool pose/tracking | Brick-to-tool centering or grasp retention |
| Hand width/SDK grasp | Mechanical contact consistent with expected width | Unique brick count, no vertical slip, source separation |
| Empty original footprint | Selected object left that region | Correct destination or intact whole pallet |
| Destination RGBD/image | Occupancy, appearance and visible height | Snapped studs, hidden contact or long-term stability |
| Unit/offline tests | Selected software contracts | Physical reliability |

## Why this choice

A single success flag hides too many possible failures. Staged trials and explicit evidence made it possible to detect a wrong orientation even after a precise release pose. Recoverable stops were preferred to silently corrupting counts or repeating an action whose outcome was unknown.

Independent work streams investigated calibration, perception/templates and manipulation; hardware commands stayed with the main orchestrator. This improved critique but did not remove the need to test the integrated physical system.

## Timing investigation: decision and evidence

1. Communication faults occurred even during holds. Analysis thread pools were capped; faults continued.
2. Host inspection found CPU scaling/idle-latency and active EthernetEEE. A reversible administrator guard set performance governors, held a zero-latency PMQoS request and disabledEEE; faults still occurred.
3. Source/process inspection found libfranka's constructor promoting its caller to FIFO 99. ROS support threads inherited that priority while the update loop was FIFO 50.
4. The local hardware adapter now saves/restores caller scheduling around construction. Runtime readback showed ordinary support-thread scheduling and one FIFO 50 update thread.
5. Subsequent 15- and 30-minute monitor intervals recorded no robot errors. This supports improved behavior but is not a controlled proof that every communication issue is resolved.

The guard and source patch address different mechanisms. The patch must be preserved when rebuilding; the guard's restoration procedure must not renegotiate Ethernet while the associated controller is active.

## Failures worth preserving

The self-avoidance reflex, endpoint-check failures, zero-finger planning state, contradictory stereo points, ambiguous shade, wrong layer orientation and off-center object are all retained in the evidence. Some failures were corrected in software and retested; others remained unresolved. The first placement must not be retroactively relabeled successful because later software supports 90-degree layers.

The many per-stage commissioning helpers also add maintenance cost. Repeated strict-TF comparison issues illustrate that shared motion/verification contracts need consolidation and regression coverage. The snapshot is a development reference, not a finished single-command demonstration.

## Remaining physical gate at this boundary

The operator reported new fingertips installed and supplied their STL. Old geometry-dependent recovery was stopped; cancellation checks found no active trajectory goals. The saved STL is an input to new geometry verification, not evidence that a new grasp was already tested.

Evidence: [15-minute timing record](../../sources/franka_lego_sorter/data/commissioning/control_timing_scheduler_restored.json), [30-minute record](../../sources/franka_lego_sorter/data/commissioning/control_timing_placement_continuation.json), [grasp/pick evidence](../../sources/franka_lego_sorter/data/commissioning/isolated_first_pick_source_proof.json). Read timestamps and associated result files rather than trusting older summary paragraphs alone.
