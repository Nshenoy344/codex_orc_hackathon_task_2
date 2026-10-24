# 04 — Brick, shade and template perception

[Algorithm, rationale, alternatives, observed outcomes and limitations](ALGORITHM_AND_DECISIONS.md).

**Decision outcome:** One12-slot reference was measured; shade overlap and body/stud ambiguity remain unresolved.

Source lives in [franka_lego_sorter/lego_sorter](../../sources/franka_lego_sorter/lego_sorter); this is a Python application using ROS 2 interfaces, not an ament package.

| Module | Role |
|---|---|
| `camera.py` | Stationary RGBD bursts, settings, intrinsics and depth units |
| `perception.py` | Robust Lab color features and distance/margin rejection |
| `pallet.py` | Brick faces, local planes, orientation and jaw-clearance checks |
| `source_slots.py` | Explicit independently traced source footprints |
| `template_registration.py` | Twelve observed faces grouped as four ridges of three |
| `scene_registration.py` | Static RGBD registration diagnostics |

The actual input has6x5 bricks. An older30-face registry had 26 labels and 4 unknowns; its positions cannot be reused after pallet movement without registration. Automatic segmentation merged touching faces, so explicit visible polygons were used rather than invented grid locations.

The orange 4x3 reference was measured after the operator removed an occluding dark brick. Seven-pixel interior masks excluded edge/background mixing without relaxing plane-fit limits. Actual nonuniform centers and orientations were retained.

Good IDs: 300121,6117418,6487946. Scrap IDs: 6135191,4211201. Labels came from operator-confirmed physical samples. The copied `data/colors.json` contains the taught model and additional tracked-maroon examples; backups/provenance are included. Lighting can make maroon and brown overlap. UNKNOWN causes reobservation/recoverable stop, never automatic scrap.

Body-top vs stud-top depth was unresolved in several observations. A small plane-fit RMS alone does not resolve that semantic distinction or absolute calibration accuracy. Isolated diagnostic grasps explicitly recorded nominal/conditional assumptions. Replacement full-height fingertips require updated physical geometry before revising those grasp constraints.

Useful artifacts: `template_reference_uncovered_interior_registration.json/png`, `first_placement_wrong_orientation_evidence.json/png`, `rotated_first_layer_camera_review.png`, and the corresponding NPZ captures under copied `data/commissioning/`.
