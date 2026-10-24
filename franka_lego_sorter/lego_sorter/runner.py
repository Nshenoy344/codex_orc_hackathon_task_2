"""Camera-driven orchestration with persistent pending-transfer accounting."""
import logging
import time
import numpy as np
from .geometry import matrix, base_detection, pick_pose, above
from .perception import detect_top_brick
from .planner import SortPlanner

LOG = logging.getLogger(__name__)


class SortRunner:
    def __init__(self, config, poses, tcp_color, model, robot, camera):
        self.c, self.e, self.robot, self.camera = config, config['execution'], robot, camera
        self.poses, self.tcp_color, self.model = poses, matrix(tcp_color), model
        self.planner = SortPlanner(config['job'], config['journal_file'])
        self.preflight()

    def preflight(self):
        if self.e.get('source_arrangement') != 'loose_stack':
            raise ValueError('Set source_arrangement=loose_stack only for unconnected bricks; connected towers need separation')
        if self.c['job'].get('source_count') is None and self.c['job'].get('requested_count') is None:
            raise ValueError('Set source_count or requested_count to bound the run')
        self.low, self.high = (np.asarray(self.e.get(k), dtype=float)
                               for k in ('workspace_min_m', 'workspace_max_m'))
        if (self.low.shape != (3,) or self.high.shape != (3,) or
                not np.isfinite(self.low).all() or not np.isfinite(self.high).all() or
                np.any(self.low >= self.high)):
            raise ValueError('Teach finite workspace_min_m/workspace_max_m')
        for name in ('consensus_frames', 'max_observation_attempts', 'max_stack_count'):
            value = self.e.get(name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        for name in ('detection_position_tolerance_m', 'source_height_tolerance_m',
                     'destination_xy_tolerance_m', 'destination_z_tolerance_m'):
            value = self.e.get(name)
            if value is None or not np.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be positive and finite')
        if not np.isfinite(self.e.get('settle_sec', -1)) or self.e['settle_sec'] < 0:
            raise ValueError('settle_sec must be finite and nonnegative')
        self.transit = float(self.e['transit_height_m'])
        if not self.low[2] < self.transit <= self.high[2]:
            raise ValueError('Transit height is outside workspace')
        if not 0 < self.e['grasp_depth_below_surface_m'] < .0096:
            raise ValueError('Grasp depth must lie inside one brick body')
        if not .009 <= self.e['stack_pitch_m'] <= .013:
            raise ValueError('Measure stack pitch (typically .0096 snapped, .0114 loose)')
        self.required_slots = {'requested', 'scrap'} | set(self.planner.leftover_routes.values())
        for name in self.required_slots | {'observation', 'pick_reference'}:
            if name not in self.poses:
                raise ValueError(f'Missing taught pose: {name}')
            self.check(self.poses[name])
        optical_down = -(matrix(self.poses['observation']) @ self.tcp_color)[2, 2]
        if optical_down < self.e.get('min_downward_optical_cosine', 0.0):
            raise ValueError('Observation camera must point approximately downward for top-surface perception')
        if self.c['perception'].get('roi') is None:
            raise ValueError('Teach source image ROI before executing')
        if self.model is None:
            raise ValueError('Teach all five color references before executing')
        # All motion, including orientation, uses the same ROS TCP as calibration.
        for name in ('observation', 'pick_reference'):
            self.poses[name] = matrix(self.poses[name])

    def check(self, pose):
        pose = matrix(pose)
        if np.any(pose[:3, 3] < self.low) or np.any(pose[:3, 3] > self.high):
            raise ValueError(f'TCP target outside measured workspace: {pose[:3, 3]}')
        return pose

    def linear(self, target):
        self.robot.move_linear(self.check(target))

    def transit_to(self, target):
        # Lift vertically before changing XY; only descend vertically at destination.
        current = self.robot.current_pose()
        self.linear(above(current, self.transit))
        self.robot.move_pose(self.check(above(target, self.transit)))
        self.linear(target)

    def observe(self, pose):
        self.transit_to(pose)
        time.sleep(self.e['settle_sec'])
        failures = []
        for _ in range(self.e['max_observation_attempts']):
            try:
                detections = [detect_top_brick(self.camera.capture_burst(), self.c['perception'], self.model)
                              for _ in range(self.e['consensus_frames'])]
                labels = {d.color_id for d in detections}
                if None in labels or len(labels) != 1:
                    raise ValueError('Color uncertain or inconsistent across frames')
                points = np.array([d.point_camera_m for d in detections])
                if np.max(np.linalg.norm(points-points.mean(axis=0), axis=1)) > self.e['detection_position_tolerance_m']:
                    raise ValueError('Brick position unstable across frames')
                actual = self.robot.current_pose()
                return detections[-1], base_detection(actual, self.tcp_color, detections[-1])
            except ValueError as exc:
                failures.append(str(exc))
        raise ValueError('Observation failed: ' + '; '.join(failures))

    def run(self, max_transfers=None):
        completed = 0
        while not self.planner.complete and (max_transfers is None or completed < max_transfers):
            det, (center, yaw) = self.observe(self.poses['observation'])
            target = pick_pose(center, yaw, self.poses['pick_reference'],
                               self.e['reference_brick_yaw_deg'], self.e['grasp_depth_below_surface_m'])
            self.check(target)
            transfer = self.planner.reserve(det.color_id)
            slot, count = transfer['destination'], transfer['stack_count_before']
            try:
                if slot != 'scrap' and count >= self.e['max_stack_count']:
                    raise ValueError(f'{slot} stack capacity reached')
                destination = matrix(self.poses[slot]).copy()
                if slot != 'scrap':
                    destination[2, 3] += count * self.e['stack_pitch_m']
                self.check(destination)
                above(target, self.transit)
                above(destination, self.transit)
                view = None
                expected_surface = destination[:3, 3].copy()
                expected_surface[2] += self.e['grasp_depth_below_surface_m']
                if self.e.get('verify_destination', True) and slot != 'scrap':
                    view = matrix(self.poses['observation']).copy()
                    view[:3, 3] += expected_surface-center
                    self.check(view)
                    above(view, self.transit)
            except Exception:
                self.planner.abort(transfer['id'])
                raise
            LOG.info('TRANSFER %s color=%s -> %s height_count=%s', transfer['id'], det.color_id, slot, count)
            self.planner.mark_motion_started(transfer['id'])
            # After this point any failure leaves a pending journal; never auto-retry a transfer.
            self.robot.open_gripper()
            self.transit_to(above(target, self.transit))
            self.linear(target)
            if not self.robot.grasp():
                raise RuntimeError('Grasp failed; pending transfer needs inspection')
            self.linear(above(target, self.transit))
            held = self.robot.gripper_state()
            if abs(held['width_m'] - self.c['motion']['grasp_width_m']) > self.c['motion']['gripper_width_tolerance_m']:
                raise RuntimeError('Held width inconsistent with one brick')
            # Look back at source while still holding the brick. The object is below
            # the camera but can occlude it; fail and inspect rather than assume removal.
            remaining = self.planner.source_remaining
            if remaining is None or remaining > 1:
                _, (after_center, _) = self.observe(self.poses['observation'])
                drop = center[2] - after_center[2]
                if abs(drop-self.e['stack_pitch_m']) > self.e['source_height_tolerance_m']:
                    raise RuntimeError(f'Source height changed {drop:.4f}m; expected one brick')
            self.transit_to(above(destination, self.transit))
            self.linear(destination)
            self.robot.open_gripper()
            self.linear(above(destination, self.transit))
            if view is not None:
                placed, (placed_center, _) = self.observe(view)
                if placed.color_id != det.color_id:
                    raise RuntimeError('Placed top color differs from picked color')
                if (np.linalg.norm(placed_center[:2]-expected_surface[:2]) > self.e['destination_xy_tolerance_m'] or
                        abs(placed_center[2]-expected_surface[2]) > self.e['destination_z_tolerance_m']):
                    raise RuntimeError('Placed brick position/height verification failed')
            self.planner.commit(transfer['id'], verified_release=True)
            completed += 1
            LOG.info('COMMITTED counts=%s', self.planner.state['counts'])
        return self.planner.state
