# Robot control and motion review — algorithm and decisions

## Method and algorithm

MoveIt computes Cartesian paths from measured arm state. The start state includes measured finger positions. The application requires a complete path, valid time parameterization, allowed joint positions/speeds and fresh state. A separate offline reviewer samples the trajectory, evaluates forward kinematics and checks the modeled arm, Hand, camera/bracket and scene geometry.

Execution uses the unchanged reviewed trajectory. The application then reads actual TCP pose and checks its distance/orientation from the intended physical endpoint. A controller's successful action result is therefore only one part of the evidence.

For nearby precision moves, the measured local tracking offset is used as a bounded feedforward correction. Translation correction is bounded by 3 mm; rotational command correction remains bounded by 10 mrad, with one partial-correction implementation capped at 9.5 mrad. Precision grasp/release poses require a separate 1 mm/10 mrad physical check before finger commands. These are application checks, not changes to the robot's collision thresholds.

## Why this choice

The installed MoveIt/ros2_control stack provides the robot model, joint planning and action interfaces. Separating planning, review and execution makes a failed stage inspectable and prevents automatically repeating a displacement after a timeout. Local compensation was chosen after actual millimetre-scale steady tracking errors appeared; one global offset would be unjustified because the bias changed with pose.

## Observed successes and failures

- Several arm actions completed but failed the independent endpoint check, with errors around 2–3.4 mm. This exposed the difference between controller completion and placement accuracy.
- Local compensation reached approximately 0.14 mm on the initial isolated grasp pose and 0.50 mm on the first release pose. These are individual trials, not a measured general accuracy guarantee.
- A folded-wrist path passed geometric review but triggered the robot's internal self-avoidance. The model did not reproduce that internal boundary. The route was revoked and conservative application wrist bounds were introduced.
- After release, MoveIt used zero finger positions despite a measured 20 mm Hand opening. State-validity queries showed false finger/brick contacts; supplying the measured finger joints made the same state valid. No collision exemption was needed.
- Some strict compensation comparisons refused plans because goal construction and planning used slightly different fresh TF samples. Selected stages introduced explicit 50 micrometre/100 microradian sampling allowances. A recovery-grasp comparison still refused before execution near the snapshot boundary; that stage should not be described as completed.

## Remaining limitations and replication conditions

Collision meshes/envelopes are approximations. They did not establish the robot's proprietary avoidance model, exact cable geometry or new fingertip shape. Saved stock-tip plans are not valid evidence for replacement tips. A successful TCP pose also does not prove that the brick is centered in the fingers or at the intended destination.

Evidence: [finger-state diagnosis](../../sources/franka_lego_sorter/data/commissioning/postrelease_start_collision_diagnostics.json), [release-pose result](../../sources/franka_lego_sorter/data/commissioning/first_loose_release_pose_01_result.json). Timing/scheduler decisions are discussed in block 06's decision guide.
