import unittest
import tempfile
from pathlib import Path
import numpy as np
from lego_sorter.camera import Frame
from lego_sorter.perception import ColorModel, detect_top_brick, color_camera_signature


class PerceptionTests(unittest.TestCase):
    def make_scene(self, color=(20, 30, 180)):
        bgr = np.full((160, 200, 3), 180, np.uint8)
        depth = np.full((160, 200), .40, np.float32)
        bgr[65:95, 70:130] = color
        depth[65:95, 70:130] = .32
        return Frame(bgr, depth, dict(fx=600., fy=600., ppx=100., ppy=80.), 0.)

    def model(self):
        model = ColorModel(max_distance=15, min_margin=5)
        for label, color in [('red', (20, 30, 180)), ('tan', (100, 160, 190)), ('brown', (35, 55, 75))]:
            patch = np.full((30, 30, 3), color, np.uint8)
            model.teach(label, patch, [2, 2, 26, 26])
        return model

    def test_known_shade_and_metric_pose(self):
        det = detect_top_brick(self.make_scene(), dict(roi=[40, 40, 120, 80]), self.model())
        self.assertEqual(det.color_id, 'red')
        np.testing.assert_allclose(det.point_camera_m, [0, 0, .32], atol=.001)
        self.assertAlmostEqual(det.diagnostics['length_m'], .0315, places=3)
        self.assertGreater(abs(det.long_axis_camera[0]), .99)

    def test_unknown_is_not_assigned(self):
        det = detect_top_brick(self.make_scene((220, 20, 20)), dict(roi=[40, 40, 120, 80]), self.model())
        self.assertIsNone(det.color_id)

    def test_invalid_depth_rejected(self):
        frame = self.make_scene()
        frame.depth_m[:] = 0
        with self.assertRaisesRegex(ValueError, 'No valid source depth'):
            detect_top_brick(frame, dict(roi=[40, 40, 120, 80]), self.model())

    def test_color_model_labels_camera_and_roundtrip(self):
        model = self.model()
        model.record_camera({'serial': 'test', 'exposure': 100})
        with self.assertRaisesRegex(ValueError, 'missing labels'):
            model.validate_labels(['red', 'unknown'])
        with self.assertRaisesRegex(ValueError, 'Camera/settings changed'):
            model.validate_camera({'serial': 'test', 'exposure': 200})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'colors.yaml'
            model.save(path)
            restored = ColorModel.load(path)
            self.assertEqual(restored.labels, {'red', 'tan', 'brown'})
            self.assertEqual(restored.metadata, model.metadata)
            self.assertTrue(restored.validate_camera({'serial': 'test', 'exposure': 100}))

    def test_depth_encoding_changes_do_not_invalidate_color_model(self):
        model=self.model()
        legacy=dict(serial='unit',camera_name='D435',width=1280,height=720,fps=30,
                    enable_auto_exposure=1.,enable_auto_white_balance=1.,exposure='auto',white_balance='auto')
        model.record_camera(legacy)
        actual=dict(legacy,depth_units_m=.0001,
                    native_depth_profile=dict(width=1280,height=720,fps=30,format='format.z16'),
                    native_color_profile=dict(width=1280,height=720,fps=30,format='format.bgr8'))
        self.assertTrue(model.validate_camera(actual))
        self.assertEqual(color_camera_signature(legacy),color_camera_signature(actual))
        self.assertEqual(model.metadata['camera'],legacy) # full original provenance retained
        for key,value in [('serial','other'),('white_balance',4200),('width',640)]:
            changed=dict(legacy);changed[key]=value
            with self.assertRaisesRegex(ValueError,'Camera/settings changed'):
                model.validate_camera(changed)
        changed=dict(actual);changed['native_color_profile']=dict(width=640,height=480,fps=30,format='format.bgr8')
        with self.assertRaisesRegex(ValueError,'Camera/settings changed'):
            model.validate_camera(changed)
        changed=dict(actual);changed['native_color_profile']=dict(width=1280,height=720,fps=30,format='format.rgb8')
        with self.assertRaisesRegex(ValueError,'Camera/settings changed'):
            model.validate_camera(changed)

    def test_wrong_size_rejected(self):
        frame = self.make_scene()
        frame.depth_m[45:115, 45:155] = .30
        with self.assertRaisesRegex(ValueError, 'matches a 2x4'):
            detect_top_brick(frame, dict(roi=[40, 40, 120, 80]), self.model())


if __name__ == '__main__':
    unittest.main()
