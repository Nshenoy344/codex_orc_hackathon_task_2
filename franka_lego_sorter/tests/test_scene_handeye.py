from pathlib import Path
import tempfile
import unittest
import cv2
import numpy as np
from lego_sorter.scene_handeye import solve_scene_handeye, calibrate_observations, CalibrationRejected, main


def pose(rv, xyz):
    p = np.eye(4)
    p[:3, :3] = cv2.Rodrigues(np.asarray(rv, float))[0]
    p[:3, 3] = xyz
    return p


def synthetic():
    rng = np.random.default_rng(421)
    x = pose([.15, -.12, .4], [.04, -.025, .10])
    g = np.array([pose(rng.normal(0, .4, 3), rng.uniform([.25, -.15, .3], [.5, .15, .55]))
                  for _ in range(18)])
    # The reference scene frame is camera0's coordinates, so target0 is identity.
    fixed = g[0] @ x
    target = np.array([np.linalg.inv(x) @ np.linalg.inv(p) @ fixed for p in g])
    return g, target, x


class SceneHandeyeTests(unittest.TestCase):
    def test_known_transform_and_heldout_validation(self):
        g, t, expected = synthetic()
        result = solve_scene_handeye(g, t)
        self.assertTrue(result['diagnostics']['accepted'])
        self.assertFalse(result['diagnostics']['physical_commissioning_verified'])
        np.testing.assert_allclose(result['T_tcp_color'], expected, atol=1e-8)
        self.assertEqual(len(result['diagnostics']['leave_one_out']), len(g))

    def test_translation_only_rejected_without_transform(self):
        g = np.array([pose([0, 0, 0], [.3, 0, .3+i*.002]) for i in range(15)])
        t = np.array([np.linalg.inv(p) @ g[0] for p in g])
        with self.assertRaisesRegex(CalibrationRejected, 'multi-axis') as context:
            solve_scene_handeye(g, t)
        self.assertFalse(context.exception.diagnostics['accepted'])
        self.assertNotIn('T_tcp_color', context.exception.diagnostics)

    def test_single_rotation_axis_rejected(self):
        g = np.array([pose([0, 0, i*.04], [.3, 0, .4]) for i in range(15)])
        t = np.array([np.linalg.inv(p) @ g[0] for p in g])
        with self.assertRaisesRegex(CalibrationRejected, 'multi-axis'):
            solve_scene_handeye(g, t)

    def test_scene_motion_rotation_mismatch_rejected(self):
        g, t, _ = synthetic()
        t[4] = pose([.3, .2, .4], [0, 0, .2]) @ t[4]
        with self.assertRaisesRegex(CalibrationRejected, 'relative rotations disagree'):
            solve_scene_handeye(g, t)

    def test_scene_translation_outlier_rejected(self):
        g, t, _ = synthetic()
        t[5, :3, 3] += [.025, 0, 0]
        with self.assertRaises(CalibrationRejected):
            solve_scene_handeye(g, t)

    def test_insufficient_actual_observations_write_diagnostic(self):
        root = Path(__file__).resolve().parents[1]
        paths = [root/'data/commissioning'/f'view_{i:02d}.npz' for i in range(3)]
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)/'result.yaml'
            code = main([*(str(p) for p in paths), '--output', str(out)])
            self.assertEqual(code, 2)
            self.assertTrue(out.is_file())
            import yaml
            data = yaml.safe_load(out.read_text())
            self.assertFalse(data['diagnostics']['accepted'])
            self.assertNotIn('T_tcp_color', data)

    def test_heldout_residual_gate_rejects(self):
        g, t, _ = synthetic()
        t[5, :3, 3] += [.001, 0, 0]
        with self.assertRaisesRegex(CalibrationRejected, 'Held-out'):
            solve_scene_handeye(g, t, dict(max_leaveout_translation_m=.0005))

    def test_solution_stability_gate_rejects(self):
        g, t, _ = synthetic()
        rng = np.random.default_rng(52)
        t[:, :3, 3] += rng.normal(0, .0002, (len(t), 3))
        with self.assertRaisesRegex(CalibrationRejected, 'unstable'):
            solve_scene_handeye(g, t, dict(max_solution_translation_spread_m=1e-8))

    def test_duplicate_files_rejected_before_image_processing(self):
        with self.assertRaisesRegex(CalibrationRejected, 'Duplicate'):
            calibrate_observations(['same.npz']*12)


if __name__ == '__main__':
    unittest.main()
