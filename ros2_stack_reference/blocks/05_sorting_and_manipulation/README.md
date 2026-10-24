# 05 — Routing, layers and manipulation

[Algorithm, rationale, alternatives, observed outcomes and limitations](ALGORITHM_AND_DECISIONS.md).

**Decision outcome:** One isolated pick succeeded; the parallel/off-center placement failed the task, and complete pallet sorting remains unproven.

Configuration: [demo.yaml](../../sources/franka_lego_sorter/config/demo.yaml). Offline reference configuration: [reference_demo.yaml](../../config/reference_demo.yaml).

Requested colors go to the right of the input. Confirmed scrap goes left. Remaining good colors go to separate upper/lower outputs. Single-color requests produce three good-color destinations; two-color requests combine requested colors and keep the third separate. Mixed requested bricks may arrive in any order; actual composition is tracked.

An output layer has four ridges of three bricks (12 slots). Fill all 12 before advancing a layer. Each destination has independent occupancy. The WHOLE measured pattern, including XY positions and orientations, rotates 90 degrees between layers. The first new layer must also be 90 degrees from the physical reference layer. `first_place_poses` describe that first new layer; subsequent rotations are relative to them. Layer pitch must be measured rather than assumed from nominal brick dimensions.

| Module | Responsibility |
|---|---|
| `planner.py` | Routing, quotas, composition and transfer state |
| `occupancy.py` | Twelve-slot geometry, rotation center, alternating layers and pitch |
| `pallet_runner.py` | Perception-to-transfer integration and verification |
| `motion.py` | ROS action adapters, fresh state, Cartesian planning and endpoint checks |
| `commissioning.py` | Explicit staged diagnostic moves |

The earlier tower `runner.py` remains for regression tests; physical pallet integration uses `PalletRunner`.

Each transfer reserves/journals before actuation. Counts and occupancy commit only after verified placement. Camera/Hand evidence is used to distinguish action completion from actual object transfer. Unknown outcomes remain pending.

Commissioning used bounded stages: open, approach, compensated grasp pose, native grasp, proof lift, carry, measured placement pose, minimal release opening and vertical retreat. Plans and independent geometry reviews are saved separately and hash-linked. Shared execution checks fresh TCP within 1 mm of the planned start. Some hardware actions completed yet failed their measured endpoint checks; those results remain failed.

The old stock fingertips imposed a narrow 2 mm usable-toe restriction. Newly installed replacement tips supersede that physical assumption only after their dimensions, TCP offset, collision shape and grasp convention are checked. The STL is preserved in `hardware/`; do not replay stock-tip commissioning trajectories against changed hardware.

Offline routing examples do not open ROS actions:

```bash
bash scripts/offline_demo.sh simulate
bash scripts/offline_demo.sh simulate --requested 300121 6117418
```

There is no demonstrated complete physical sorting launch in this snapshot. Missing measured station templates/poses, layer pitch and physical calibration approval must remain missing rather than be filled with guessed values.
