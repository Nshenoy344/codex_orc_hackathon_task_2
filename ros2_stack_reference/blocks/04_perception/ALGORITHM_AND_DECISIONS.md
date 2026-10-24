# Brick geometry, shade classification and templates — algorithm and decisions

## Method and algorithm

Brick proposals use image/color segmentation, visible contours and aligned depth. Local surface planes and deprojected corners estimate metric footprint, long axis and height. The source registry also supports independently traced face polygons where segmentation merges adjacent bricks. A known arrangement is not used to invent an unseen or occluded brick.

Template registration requires 12 independently observed footprints. It estimates a common orientation, projects centers along/across the ridges, groups four ridges of three and checks dimensions, alignment, overlap and support-height consistency. Every measured position is retained rather than snapped onto a perfect grid.

Shade classification is a small supervised prototype classifier:

1. Select a physical brick's interior pixels and convert BGR to OpenCV Lab.
2. Remove the darkest/brightest 15% of selected luminance values.
3. Take the remaining median Lab vector.
4. Calculate each label's nearest Euclidean prototype distance.
5. Accept only if the best distance is at most 18 and its margin over the second label is at least 5; otherwise returnUNKNOWN.

Those thresholds are in OpenCV's Lab encoding. They are not calibrated probabilities or a spectrophotometer's color measurements.

## Why this choice

Five known physical shades and a hackathon deadline made classical geometry and operator-taught color prototypes a practical initial choice. A large neural model/dataset was not available or required to begin testing. Interior statistics reduce reflections and shadows; multiple physical examples cover some view variation.

Explicit polygons were a commissioning bootstrap when touching faces merged. This avoided fabricating grid coordinates but introduced manual work; it does not prove autonomous detection of every future pallet.

## Observed successes and failures

- The observed input was a 6x5 pallet, not the initially assumed tower. The tower abstraction was unsuitable and the pallet path replaced it.
- The original 30-face source record yielded 26 labels and 4 unknowns. Pallet movement invalidated its old coordinates; this was not solved by continuing to reuse the old pixels.
- A dark brick occluded two output-reference faces. Removing it exposed all 12; the algorithm did not fill missing corners from an ideal grid.
- Edge/background depth mixing made right-ridge plane fits fail. A 7-pixel interior mask improved observed face fits to roughly 0.44–0.67 mm RMS while retaining the 3 mm limit.
- A small-face fit could have low residual yet suggest an implausible 10-degree normal. Recovery geometry used an independently measured broad support normal, keeping the local normal only as a diagnostic.
- Maroon and brown teaching samples overlapped under different lighting. Adding a provenance-backed maroon example did not make them reliably separable and increased ambiguity in one brown example. UNKNOWN was preserved rather than relaxing the threshold.

## Remaining limitations and alternatives

Body-top and stud-top depth remained unresolved in several views. A clean plane is not automatically the correct grasp surface. The original stock fingers had a very small insertion budget, making that distinction important. Replacement full-height tips may improve tolerance, but require measured geometry and physical tests.

Controlled lighting, fixed validated color settings, illumination normalization or a better supervised classifier are possible next steps; they are not demonstrated solutions in this snapshot. Likewise, manual 12-slot registration is established for one reference view, not automatic robust detection of arbitrary templates.

Evidence: [measured 12-face template](../../sources/franka_lego_sorter/data/commissioning/template_reference_uncovered_interior_registration.json), [wrong-placement/color evidence](../../sources/franka_lego_sorter/data/commissioning/first_placement_wrong_orientation_evidence.json).
