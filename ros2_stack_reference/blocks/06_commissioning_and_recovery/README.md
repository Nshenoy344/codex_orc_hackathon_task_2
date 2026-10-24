# 06 — Physical evidence, recovery and logs

[Algorithm, rationale, alternatives, observed outcomes and limitations](ALGORITHM_AND_DECISIONS.md).

**Decision outcome:** Separate evidence exposed failures; monitored timing improved after a scheduling fix, without proving general reliability.

The copied `sources/franka_lego_sorter/data/` and `logs/` preserve runtime evidence that Git normally excludes. Upstream README/status files may describe earlier stages; dated action results and images must be interpreted together.

## Established physical results

One operator-confirmed 6117418 maroon brick was picked from an isolated white-table position at 15 N. Jaw width stayed near 15.35 mm through a 9.75 mm proof lift and a larger raise. A subsequent RGBD capture found its original footprint matching the measured floor:99.9% valid depth, median absolute floor residual 0.54 mm.

A compensated placement pose reached 0.50 mm measured position error; a 20.05 mm Hand opening released the brick. Later images showed it parallel to the reference ridges and off center. This is **not a correct task placement** and no production occupancy/count was committed. Input-pallet individual extraction, completed layers, repeated-layer pitch and both full request modes remain undemonstrated.

Key evidence files include `isolated_first_grasp.json`, `isolated_first_lift10_result.json`, `isolated_first_pick_source_proof.json`, `first_loose_release_pose_01_result.json`, `first_loose_release_01.json`, and `first_placement_wrong_orientation_evidence.json`.

## Failures and fixes

- A folded-wrist move triggered internal self-avoidance despite geometric planning checks. The route was revoked. Application wrist bounds were added; robot thresholds were not relaxed.
- Communication faults persisted after analysis-thread limits and an administrator latency guard. The source-level constructor scheduling restoration then prevented FIFO 99 leakage into support threads. Readback showed support threads OTHER 0 and the update loop FIFO 50. Subsequent 15- and 30-minute monitor intervals recorded no robot errors; this does not guarantee reliability.
- MoveIt had zero finger joint positions despite actual 20 mm opening. Planning now seeds measured Hand positions. State-validity contacts diagnosed the false collision before any retreat action.
- Three dark-gap stereo points blocked recovery. A new supported HighAccuracy preset 3 capture removed them; no collision gate was bypassed.
- Some trajectories physically completed but failed measured endpoint tolerance. Subsequent observation used actual pose and retained the original failure, rather than replaying a displacement or pretending success.

## Journals and recovery

Inspect pending state and actual images before retrying any failed grasp/release. A native action may have occurred even when verification fails. Unique attempt files and exclusive markers prevent automatic replay. Reconciliation records an inspected physical outcome; it is not a robot command.

The temporary root latency guard must remain active during the associated controller session. Cleanup order: stop arm/controller first, confirm it exited, then interrupt the guard so CPU/EEE settings restore. Never restore Ethernet EEE underneath an active controller. See the copied application README and `tools/robot_host_performance.py` for the saved-state mechanism.

Replacement fingertips were reported installed at the boundary. Their STL is in `hardware/franka_research3_gripper.stl`. All stock-finger pose/clearance assumptions need review before further descents. The side-reference export itself did not touch robot processes or devices.

## Logs

```bash
tail -f logs/commissioning.log
tail -f logs/demo.log
```

These relative commands refer to the application directory. The bundle contains saved copies; they are not live-following links. For current events, use the original working application's logs. Per-attempt planning/validation/execution logs are under its `logs/`, with JSON results in `data/commissioning/`.
