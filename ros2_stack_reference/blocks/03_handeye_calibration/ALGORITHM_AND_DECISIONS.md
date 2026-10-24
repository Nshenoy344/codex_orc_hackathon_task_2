# Camera-to-TCP estimation — algorithm and decisions

## Method and algorithm

For a stationary board and wrist-mounted camera, each capture supplies robot pose `G_i=T_base_tcp_i` and board pose `C_i=T_color_board_i`. Solve for `X=T_tcp_color` such that:

```text
G_i @ X @ C_i = constant base-to-board pose
```

Relative observations yield the hand-eye equation `A X = X B`; the recorded solution used the PARK method. Distinct rotations about multiple axes are needed for observability, not just many similar images.

Board pose was first attempted using image-corner PnP. The accepted numerical candidate instead used the `rgbd_plane` estimator: detect identified ChArUco corners, fit a board depth plane, intersect corner rays with it, fit the fixed metric board points to those observed 3D corners, and solve/validate hand-eye geometry. The rigid corner fit uses Kabsch/SVD; the subsequent hand-eye solve is a different step.

Validation checks board consistency in the base frame, rotation excitation, held-out residuals and solution stability. `T_tcp_depth` is composed with the SDK factory depth-to-color transform; its inverse maps TCP coordinates into native depth coordinates.

## Why this choice

ChArUco supplies identifiable subpixel image corners under partial visibility. A stationary board removes the need for an independent tracker. The depth-plane constraint was introduced after PnP-only target poses failed downstream consistency checks; it uses actual depth rather than accepting a good-looking image reprojection score as sufficient evidence.

Early near-translation-only scene observations were rejected because they did not sufficiently constrain all transform components. Collecting diverse measured orientations was preferred to exporting an arbitrary matrix from insufficient data.

## Observed successes and failures

- Seventeen measured views passed the numerical RGBD-plane validation: approximately 0.33 mm training translation RMS, 0.85 mm maximum held-out residual and 1.71 mm leave-one-out translation spread.
- A PnP-only candidate and insufficient-excitation observations were not accepted as equivalent successes.
- The actual board was 7x5 squares with 4x4-family markers; example package settings were different. Using the example YAML unchanged would describe another board.
- Square size about 29.95 mm came from aligned-depth geometry. Its physical print scale was not independently measured, so low residuals do not certify absolute scale.

## Remaining limitations and replication conditions

`physical_commissioning_verified` remains false. Numerical fit is not a physical pick/placement certificate. Mount movement, changed TCP conventions, wrong board dictionary/scale, inconsistent timestamps or mixed native/aligned depth frames invalidate the result.

The ROS calibration package does not command robot motion. Its default PnP workflow and the sorter's RGBD-plane commissioning estimator are separate choices; reproduce the latter with `--estimator rgbd_plane` and the actual saved board configuration.

Evidence: [numerical candidate and diagnostics](../../sources/franka_lego_sorter/data/commissioning/charuco_rgbd_plane_candidate.yaml). The original paired captures and board configuration are saved beside it.
