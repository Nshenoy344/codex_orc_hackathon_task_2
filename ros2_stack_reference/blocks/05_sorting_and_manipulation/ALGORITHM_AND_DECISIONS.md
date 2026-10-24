# Routing, alternating layers and grasp strategy — algorithm and decisions

## Method and algorithm

Routing is deterministic: confidently identified rejects go left; requested good colors go right; other good colors map to upper/lower destinations. A two-color request combines those colors in arrival order and records their actual composition. `requested_count` is a total requested-output quota, not a per-color quota. Ordered sequences are not implemented.

For committed brick count `n`, a destination's next layer is`n//12` and next slot is`n%12`. The first-new-layer poses are already rotated 90 degrees from the physical reference. Subsequent layer `k` rotates the complete measured pattern around measured center`c`:

```text
p_i,k = c + Rz(90deg*k) @ (p_i,0 - c) + [0,0,k*h]
R_i,k = Rz(90deg*k) @ R_i,0
```

`h` is a measured pitch, not an assumed LEGO dimension. This implementation grows layers along baseZ; a sloped stacking fixture would need that convention revisited.

Manipulation commissioning used open -> approach -> compensated grasp pose -> native grasp -> proof lift -> carry -> measured release pose -> minimal opening -> retreat -> visual verification. A pending record precedes an actuator command, and occupancy commits only after verification. Unique-attempt markers prevent an uncertain release being repeated under another output filename.

## Why this choice

Explicit 12-slot geometry preserves the user's ridge layout. Rotating the full pattern preserves its spacing; rotating only each brick in place would not produce the intended alternating layer. Journaling is needed because the robot may act successfully before the process crashes or verification fails.

Isolated-brick tests were selected before dense-pallet extraction to distinguish basic grasp feasibility from neighbor clearance and brick attachment. A shallow stud grasp was investigated as an alternative to the stock finger's screw-clearance limitation, but was not demonstrated.

## Observed successes and failures

- Offline routing tests exercise one- and two-color requests, composition,12-slot progression and recovery. This is software evidence only.
- The isolated maroon body grasp at 15 N retained about 15.35 mm jaw width through a 9.75 mm proof lift. The original footprint then matched the measured floor, supporting a real isolated pick.
- The first release physically occurred, but the brick was parallel to the underlying ridges and slightly off center. This failed the intended task. Precise TCP feedback did not establish correct object orientation or centering.
- The initial placement plan omitted the required alternating orientation. Whole-pattern 90-degree logic was added afterward; its physical correction was not complete at the snapshot boundary.
- Rotating the measured nonsquare pattern exposed weak outer-edge support. The geometric support-hull margin for proposed outer slot r3b2 was only about 0.79 mm; middle slot r1b2 offered about 8.26 mm. Selecting the middle slot was a proposed correction, not a completed placement. Footprint overlap does not prove stud engagement or mechanical stability.

## Remaining limitations and replication conditions

No actual input-pallet extraction, correctly completed 12-slot layer, subsequent physical layer or full single/two-color run is established. Whether bricks are attached and whether a base clamp/seating method is needed must be learned through physical tests.

Original isolated grasps sometimes used explicitly nominal dimensions or a bottom plane conditional on flat support/no slip. They were not measurements of the hidden body surface or a universal object-to-TCP transform. New tips invalidate old toe/screw envelopes; their STL is saved but physical commissioning remains pending.

A robust complete interlocking pallet may require validated stud seating or extra support for edge bricks. Pure geometric alignment and a small release gap do not prove that result.

Evidence: [isolated pick proof](../../sources/franka_lego_sorter/data/commissioning/isolated_first_pick_source_proof.json), [failed task placement](../../sources/franka_lego_sorter/data/commissioning/first_placement_wrong_orientation_evidence.json), [rotated pattern](../../sources/franka_lego_sorter/data/commissioning/rotated_first_layer_pattern.json).
