# Environment and process isolation — algorithm and decisions

## Method and algorithm

Use one compatible Python/ROS environment: system Python, system OpenCV/NumPy/cv_bridge, ROS 2 Humble and the installed Franka overlay. Set `PYTHONNOUSERSITE=1`, use ROS domain 31, prepend local modules without replacing ROS's PYTHONPATH, and cap analysis thread pools. The saved RealSense SDK runtime supplies the direct camera bindings.

This block is dependency and process management rather than a computer-vision algorithm. Its output is a predictable import environment and ROS graph in which later algorithms can run.

## Why this choice

The existing robot stack already worked with ROS Humble. Reusing it avoided introducing another control protocol and a second set of frame conventions. Keeping the SDK in a separate runtime directory avoided replacing the system OpenCV installation. Explicit domain selection prevented discovery failures caused by an inherited domain 23.

The alternative of installing newer NumPy/OpenCV packages globally was not used because it could break ROS's binary Python extensions. Analysis thread limits were selected to reduce CPU contention alongside the real-time controller; they do not by themselves make a host real-time capable.

## Observed successes and failures

- User-site NumPy 2 conflicted with system OpenCV's ABI. System Python with user-site packages disabled allowed the capture/perception code to run.
- Explicit ROS domain 31 made the application use the intended robot graph rather than the unrelated inherited domain.
- Thread limits alone did **not** eliminate robot communication faults. That hypothesis was insufficient; later scheduling inspection found another concrete problem.

## Remaining limitations and replication conditions

Reproduce the recorded dependency versions or test a deliberate upgrade. The saved SDK binary is not portable to arbitrary CPU architectures/Python ABIs. Fresh ROS discovery can take time; bounded discovery waits are appropriate, but missing state must not be substituted with cached guesses.

See [dependency versions](../../provenance/system_packages.tsv), [host record](../../provenance/host.txt) and the environment wrapper. No host scheduling change is applied by sourcing that wrapper.
