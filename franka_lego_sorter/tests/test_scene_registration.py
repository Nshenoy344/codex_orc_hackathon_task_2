import unittest
import cv2
import numpy as np
from lego_sorter.scene_registration import estimate_rigid_ransac, RegistrationError, register_rgbd
from lego_sorter.camera import Frame


class RegistrationTests(unittest.TestCase):
    def test_known_transform_with_outliers(self):
        rng = np.random.default_rng(6)
        reference = rng.uniform([-.10, -.08, .3], [.10, .08, .34], (100, 3))
        rotation = cv2.Rodrigues(np.array([.08, -.15, .12]))[0]
        translation = np.array([.025, -.018, .035])
        current = reference@rotation.T+translation+rng.normal(0, .0004, reference.shape)
        current[::4] = rng.uniform(-.2, .4, (25, 3))
        result = estimate_rigid_ransac(reference, current)
        np.testing.assert_allclose(result.T_current_reference[:3, :3], rotation, atol=.002)
        np.testing.assert_allclose(result.T_current_reference[:3, 3], translation, atol=.001)
        self.assertEqual(result.metrics['inlier_count'], 75)
        self.assertLess(result.metrics['residual_rms_m'], .001)

    def test_two_coherent_displacements_rejected_as_ambiguous(self):
        rng = np.random.default_rng(29)
        reference = rng.uniform([-.10, -.08, .3], [.10, .08, .34], (60, 3))
        current = reference.copy()
        current[:30] += [.04, 0, 0]
        with self.assertRaisesRegex(RegistrationError, 'Ambiguous scene'):
            estimate_rigid_ransac(reference, current)

    def test_collinear_geometry_rejected(self):
        line = np.column_stack([np.linspace(-.1, .1, 40), np.zeros(40), np.full(40, .3)])
        with self.assertRaises(RegistrationError):
            estimate_rigid_ransac(line, line)

    def test_small_repeated_stud_patch_rejected(self):
        rng = np.random.default_rng(1)
        points = rng.uniform([0, 0, .3], [.01, .01, .301], (40, 3))
        with self.assertRaisesRegex(RegistrationError, 'spread|small'):
            estimate_rigid_ransac(points, points)

    def test_planar_target_is_supported(self):
        rng = np.random.default_rng(5)
        points = np.column_stack([rng.uniform(-.1, .1, (60, 2)), np.full(60, .3)])
        result = estimate_rigid_ransac(points, points+np.array([.02, 0, .01]))
        self.assertEqual(result.metrics['reference_spread']['rank'], 2)
        np.testing.assert_allclose(result.T_current_reference[:3, 3], [.02, 0, .01], atol=1e-8)

    def test_insufficient_inlier_count_rejected(self):
        rng = np.random.default_rng(2)
        points = rng.uniform(-.1, .1, (19, 3))
        with self.assertRaisesRegex(RegistrationError, 'at least 20'):
            estimate_rigid_ransac(points, points)

    def test_untextured_image_is_not_success(self):
        frame = Frame(np.full((200, 240, 3), 100, np.uint8), np.full((200, 240), .3),
                      dict(fx=600., fy=600., ppx=120., ppy=100.), 0.)
        with self.assertRaisesRegex(RegistrationError, 'Too few textured'):
            register_rgbd(frame, frame)


if __name__ == '__main__':
    unittest.main()
