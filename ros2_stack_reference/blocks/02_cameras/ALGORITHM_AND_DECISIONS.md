# RGBD acquisition and auxiliary observation — algorithm and decisions

## Method and algorithm

The commissioning path opens one RealSense SDK pipeline, captures RGB and depth, aligns depth into color pixels, rejects invalid zero depth and takes a per-pixel median across a stationary burst. Intrinsics, actual sensor settings, depth scale and factory color/depth extrinsics accompany the image. ROS TF is sampled before/after acquisition; excessive measured movement rejects the capture.

Deprojection turns an aligned pixel and its depth into a color-optical-frame 3D point. Robot-base coordinates follow from `T_base_tcp @ T_tcp_color`. Auxiliary webcams provide additional pictures of the gripper/pallet. They have no verified metric robot extrinsics in this snapshot.

The alternative ROS capture path in the hand-eye package reads image/CameraInfo topics and timestamped TF. It exists separately; the physical sorter commissioning described here used direct SDK captures with stationary pose pairing.

## Why this choice

Depth provides metric scene information, while RGB provides the brick boundaries and shade information. Median bursts suppress transient noise and retain raw evidence for offline reprocessing. Stationary acquisition reduces synchronization ambiguity; it is not a substitute for exact timestamp handling during moving-camera perception.

Direct SDK capture exposed factory extrinsics and sensor controls without requiring a competing camera ROS driver. Using auxiliary views was helpful for detecting failures that joint feedback alone cannot reveal. Their images were not promoted into metric measurements without calibration.

## Observed successes and failures

- RealSense Viewer occupied the stream and prevented another consumer from capturing. Closing it resolved that access conflict.
- At roughly 259 mm range, close-range brick depth was unreliable; local floor fits could look good while the dark brick surface had missing or misleading depth.
- Finer 100 micrometre depth encoding did not resolve the body/stud surface problem. Encoding resolution was not sensor accuracy; the experiment was not treated as a successful fix.
- Three dark-gap points appeared near world Z=70 mm although the visible structures were lower. They blocked a recovery approach before motion.
- Reobserving with supported High Accuracy preset 3 removed those three points in the inspected region. Global valid depth decreased from about 99.2% to 93.3%; the new maroon interior fit used 1,590 valid pixels and had 0.356 mm RMS. This was an observed tradeoff, not permission to fill missing depth.
- A preset write while streams were active failed with a device-busy error. Configuring it before stream startup succeeded.

## Remaining limitations and replication conditions

Repeated LEGO texture, dark gaps, specular studs, occlusions and insufficient range can all corrupt stereo measurements. Reobserve rather than assume all nonzero depth is valid. High Accuracy was useful in one comparison; it is not guaranteed to remove every false match. Plane RMS measures internal fit, not absolute accuracy.

Auxiliary cameras were moved/added during commissioning. Device role and display rotation are recorded, but old camera geometry cannot be reused after repositioning. Keep a single RealSense stream owner and record actual settings for every metric observation.

Evidence: [rejected-gap points](../../sources/franka_lego_sorter/data/commissioning/recovery_approach_collision_pixels.json), [High Accuracy observation](../../sources/franka_lego_sorter/data/commissioning/recovery_depth_high_accuracy_02.npz).
