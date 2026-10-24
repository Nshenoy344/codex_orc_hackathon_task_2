import unittest
import cv2
import numpy as np
from franka_realsense_handeye.core import solve, transform
from franka_realsense_handeye.board import make_board, detect


class CalibrationTest(unittest.TestCase):
    def test_known_handeye_and_depth_direction(self):
        rng = np.random.default_rng(7)
        x = transform(cv2.Rodrigues(np.array([.15, -.2, .3]))[0], [.03, -.05, .11])
        fixed = transform(np.eye(3), [.5, .1, .2])
        robots = [transform(cv2.Rodrigues(rng.normal(0, .5, 3))[0],
                            rng.uniform(-.3, .3, 3)) for _ in range(20)]
        targets = [np.linalg.inv(x) @ np.linalg.inv(g) @ fixed for g in robots]
        for method in ('TSAI', 'PARK', 'HORAUD', 'ANDREFF', 'DANIILIDIS'):
            actual, metrics = solve(robots, targets, method)
            np.testing.assert_allclose(actual, x, atol=1e-6)
            self.assertLess(metrics['board_translation_rms_m'], 1e-6)
        depth = transform(cv2.Rodrigues(np.array([.01, .02, -.01]))[0], [.015, 0, 0])
        point = np.array([.1, .2, 1., 1.])
        np.testing.assert_allclose(np.linalg.inv(actual @ depth) @ (x @ depth @ point), point)

    def test_degenerate_rotations_rejected(self):
        robots = [transform(cv2.Rodrigues(np.array([0., 0., i*.1]))[0], [i*.01, 0, 0])
                  for i in range(15)]
        with self.assertRaisesRegex(ValueError, 'rotation diversity'):
            solve(robots, robots)

    def test_insufficient_samples_rejected(self):
        with self.assertRaisesRegex(ValueError, 'paired samples'):
            solve([np.eye(4)], [np.eye(4)])

    def test_board_image_pose(self):
        p = dict(dictionary='DICT_5X5_1000', squares_x=12, squares_y=7,
                 square_length_m=.04, marker_length_m=.03, legacy_pattern=False)
        board, dictionary = make_board(p)
        image = (board.generateImage((1200, 700)) if hasattr(board, 'generateImage')
                 else board.draw((1200, 700)))
        image = cv2.copyMakeBorder(image, 100, 100, 100, 100, cv2.BORDER_CONSTANT, value=255)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        k = np.array([[1000., 0, 700], [0, 1000., 450], [0, 0, 1.]])
        _, tv, rms, count = detect(image, k, np.zeros(5), board, dictionary, 20)
        self.assertGreater(count, 20)
        self.assertLess(rms, 1.)
        self.assertAlmostEqual(float(tv[2, 0]), .4, places=2)


if __name__ == '__main__':
    unittest.main()
