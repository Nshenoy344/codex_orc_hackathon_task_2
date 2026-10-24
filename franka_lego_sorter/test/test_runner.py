from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from lego_sorter.runner import SortRunner


def pose(x=0., y=0., z=0.):
    p = np.eye(4)
    p[:3, 3] = [x, y, z]
    return p


def detection(color='red', z=.2):
    return SimpleNamespace(color_id=color, point_camera_m=np.array([0., 0., z]),
                           long_axis_camera=np.array([1., 0., 0.]))


class FakeRobot:
    def __init__(self, grasp_ok=True):
        self.position = pose()
        self.events = []
        self.grasp_ok = grasp_ok

    def current_pose(self):
        return self.position.copy()

    def move_linear(self, target):
        self.events.append(('linear', target.copy()))
        self.position = target.copy()

    def move_pose(self, target):
        self.events.append(('pose', target.copy()))
        self.position = target.copy()

    def open_gripper(self):
        self.events.append(('open', None))

    def grasp(self):
        self.events.append(('grasp', None))
        return self.grasp_ok

    def gripper_state(self):
        return {'width_m': .0158}


class FakeCamera:
    def capture_burst(self):
        return object()


class RunnerTest(unittest.TestCase):
    def config(self, path, source_count=2):
        return dict(job=dict(good_colors=['red', 'dark', 'orange'], bad_colors=['tan', 'brown'],
                             requested_colors=['red'], source_count=source_count,
                             requested_count=1), journal_file=str(path),
                    perception=dict(roi=[0, 0, 100, 100]),
                    motion=dict(grasp_width_m=.0158, gripper_width_tolerance_m=.003),
                    execution=dict(source_arrangement='loose_stack',
                                   min_downward_optical_cosine=-1., # mocked camera geometry

                                   workspace_min_m=[-.5, -.5, -.1], workspace_max_m=[.5, .5, .6],
                                   transit_height_m=.5, grasp_depth_below_surface_m=.006,
                                   stack_pitch_m=.0114, reference_brick_yaw_deg=0.,
                                   max_stack_count=12, settle_sec=0., consensus_frames=1,
                                   max_observation_attempts=1, detection_position_tolerance_m=.004,
                                   source_height_tolerance_m=.004, verify_destination=True,
                                   destination_xy_tolerance_m=.006, destination_z_tolerance_m=.006))

    def poses(self):
        return dict(observation=pose(), pick_reference=pose(), requested=pose(.1, 0., .194),
                    scrap=pose(-.1, 0., .194), upper=pose(0., .1, .194), lower=pose(0., -.1, .194))

    def test_one_transfer_verifies_source_and_destination_before_commit(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'run.json'
            robot = FakeRobot()
            runner = SortRunner(self.config(path), self.poses(), np.eye(4), object(), robot, FakeCamera())
            with patch('lego_sorter.runner.detect_top_brick', side_effect=[
                    detection(), detection('dark', .1886), detection()]) as detect:
                state = runner.run(max_transfers=1)
            self.assertEqual(detect.call_count, 3)
            self.assertEqual(state['counts']['requested'], 1)
            self.assertIsNone(state['pending'])
            self.assertEqual(sum(e[0] == 'open' for e in robot.events), 2)
            self.assertEqual(json.loads(path.read_text())['processed'], 1)

    def test_unknown_source_count_still_checks_one_brick_removed(self):
        with tempfile.TemporaryDirectory() as d:
            runner = SortRunner(self.config(Path(d)/'run.json', source_count=None), self.poses(),
                                np.eye(4), object(), FakeRobot(), FakeCamera())
            with patch('lego_sorter.runner.detect_top_brick', side_effect=[
                    detection(), detection('dark', .1886), detection()]) as detect:
                runner.run(max_transfers=1)
            self.assertEqual(detect.call_count, 3)

    def test_wrong_source_height_leaves_pending_and_holds_brick(self):
        with tempfile.TemporaryDirectory() as d:
            robot = FakeRobot()
            runner = SortRunner(self.config(Path(d)/'run.json'), self.poses(),
                                np.eye(4), object(), robot, FakeCamera())
            with patch('lego_sorter.runner.detect_top_brick', side_effect=[
                    detection(), detection('dark', .1772)]):
                with self.assertRaisesRegex(RuntimeError, 'Source height'):
                    runner.run(max_transfers=1)
            self.assertIsNotNone(runner.planner.state['pending'])
            self.assertEqual(runner.planner.state['processed'], 0)
            self.assertEqual(sum(e[0] == 'open' for e in robot.events), 1)

    def test_failed_grasp_does_not_commit_or_retry(self):
        with tempfile.TemporaryDirectory() as d:
            robot = FakeRobot(grasp_ok=False)
            runner = SortRunner(self.config(Path(d)/'run.json'), self.poses(),
                                np.eye(4), object(), robot, FakeCamera())
            with patch('lego_sorter.runner.detect_top_brick', return_value=detection()):
                with self.assertRaisesRegex(RuntimeError, 'Grasp failed'):
                    runner.run(max_transfers=1)
            self.assertTrue(runner.planner.state['pending']['motion_started'])
            self.assertEqual(runner.planner.state['processed'], 0)

    def test_known_last_brick_skips_source_view(self):
        with tempfile.TemporaryDirectory() as d:
            runner = SortRunner(self.config(Path(d)/'run.json', source_count=1), self.poses(),
                                np.eye(4), object(), FakeRobot(), FakeCamera())
            with patch('lego_sorter.runner.detect_top_brick', side_effect=[detection(), detection()]) as detect:
                runner.run(max_transfers=1)
            self.assertEqual(detect.call_count, 2)

    def test_invalid_consensus_rejected_without_motion(self):
        with tempfile.TemporaryDirectory() as d:
            config = self.config(Path(d)/'run.json')
            config['execution']['consensus_frames'] = 0
            robot = FakeRobot()
            with self.assertRaisesRegex(ValueError, 'consensus_frames'):
                SortRunner(config, self.poses(), np.eye(4), object(), robot, FakeCamera())
            self.assertEqual(robot.events, [])


if __name__ == '__main__':
    unittest.main()
