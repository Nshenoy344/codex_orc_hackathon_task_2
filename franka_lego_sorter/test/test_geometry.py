import unittest
from types import SimpleNamespace
import numpy as np
from lego_sorter.geometry import base_detection, pick_pose, above


class GeometryTest(unittest.TestCase):
    def test_camera_extrinsic_composition_is_not_inverted(self):
        robot = np.eye(4)
        robot[:3, 3] = [.4, .1, .5]
        camera = np.eye(4)
        camera[:3, :3] = np.diag([1., -1., -1.])
        camera[:3, 3] = [.03, 0, .02]
        d = SimpleNamespace(point_camera_m=[.02, .01, .3], long_axis_camera=[1, 0, 0])
        p, yaw = base_detection(robot, camera, d)
        np.testing.assert_allclose(p, [.45, .09, .22])
        self.assertAlmostEqual(yaw, 0.)

    def test_undirected_brick_axis_does_not_flip_gripper(self):
        ref = np.eye(4)
        a = pick_pose([.4, .1, .2], 0., ref, 0., .006)
        b = pick_pose([.4, .1, .2], np.pi, ref, 0., .006)
        np.testing.assert_allclose(a, b, atol=1e-10)
        self.assertAlmostEqual(a[2, 3], .194)

    def test_transit_below_target_refused(self):
        t = np.eye(4)
        t[2, 3] = .5
        with self.assertRaises(ValueError):
            above(t, .4)

    def test_vertical_axis_refused(self):
        d = SimpleNamespace(point_camera_m=[0, 0, .3], long_axis_camera=[0, 0, 1])
        with self.assertRaises(ValueError):
            base_detection(np.eye(4), np.eye(4), d)


if __name__ == '__main__':
    unittest.main()
