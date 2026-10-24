import unittest
import numpy as np
from lego_sorter.camera import Frame
from lego_sorter.template_registration import group_measured_slots, register_template, TemplateError


class TemplateTests(unittest.TestCase):
    def observations(self):
        result = []
        # Deliberately irregular gaps, with differently staggered longitudinal centers.
        for ridge, x in enumerate([-.070, -.035, .014, .066]):
            for y in [-.050+ridge*.001, -.012+ridge*.001, .034+ridge*.001]:
                result.append(dict(point_camera_m=[x, y, .45], long_axis_camera=[0, 1, 0],
                                   surface_normal_camera=[0, 0, -1], length_m=.0318, width_m=.0158,
                                   center_px=[100+x*1000, 100+y*1000], box_px=[], surface_plane_rms_m=.0005))
        return result

    def fixture(self):
        color = np.full((600, 800, 3), 150, np.uint8)
        depth = np.full((600, 800), .50, np.float32)
        polygons = []
        for x in [200, 300, 420, 560]:
            for y in [110, 260, 430]:
                polygon = [[x-32, y-64], [x+32, y-64], [x+32, y+64], [x-32, y+64]]
                depth[y-64:y+65, x-32:x+33] = .45
                polygons.append(polygon)
        frame = Frame(color, depth, dict(fx=1800., fy=1800., ppx=400., ppy=300.), 0.)
        config = dict(template_type='fixture', template_type_verified=True,
                      station_polygon_px=[[150, 40], [610, 40], [610, 510], [150, 510]],
                      slot_polygons_px=polygons, table_polygon_px=[[10, 10], [100, 10], [100, 550], [10, 550]])
        return frame, config

    def test_measured_irregular_gaps_and_every_center_preserved(self):
        observations = self.observations()
        result = group_measured_slots(observations, dict(template_type='brick_layer'))
        self.assertEqual(len(result.slots), 12)
        np.testing.assert_allclose(result.ridge_gaps_m, [.0192, .0332, .0362], atol=1e-9)
        np.testing.assert_allclose(result.within_ridge_gaps_m[0], [.0062, .0142], atol=1e-9)
        np.testing.assert_allclose([s.point_camera_m for s in result.slots], [x['point_camera_m'] for x in observations])
        self.assertFalse(result.placement_ready)
        self.assertIsNone(result.support_height_above_table_m)

    def test_fixture_has_measured_height_and_twelve_slots(self):
        frame, config = self.fixture()
        result = register_template(frame, config)
        self.assertTrue(result.placement_ready)
        self.assertEqual(result.template_type, 'fixture')
        self.assertAlmostEqual(result.support_height_above_table_m, .05, places=5)
        self.assertEqual(len(result.slots), 12)
        self.assertFalse(result.metrics['vertical_layer_increment_measured'])
        self.assertNotAlmostEqual(result.ridge_gaps_m[0], result.ridge_gaps_m[-1], places=3)

    def test_base_type_is_explicit_not_inferred_as_brick_layer(self):
        frame, config = self.fixture()
        config['template_type'] = 'base'
        result = register_template(frame, config)
        self.assertEqual(result.template_type, 'base')
        config['template_type_verified'] = False
        self.assertFalse(register_template(frame, config).placement_ready)

    def test_missing_geometry_is_not_invented(self):
        with self.assertRaisesRegex(TemplateError, 'Exactly 12'):
            group_measured_slots(self.observations()[:-1], dict(template_type='brick_layer'))
        frame, config = self.fixture()
        del config['slot_polygons_px']
        with self.assertRaisesRegex(TemplateError, '12 individually observed'):
            register_template(frame, config)

    def test_nonparallel_or_overlapping_geometry_rejected(self):
        observations = self.observations()
        observations[4]['long_axis_camera'] = [1, 0, 0]
        with self.assertRaisesRegex(TemplateError, 'parallel ridges'):
            group_measured_slots(observations, dict(template_type='brick_layer'))
        observations = self.observations()
        observations[2]['point_camera_m'] = list(observations[1]['point_camera_m'])
        with self.assertRaisesRegex(TemplateError, 'overlap'):
            group_measured_slots(observations, dict(template_type='brick_layer'))

    def test_unknown_template_type_and_invalid_depth_rejected(self):
        with self.assertRaisesRegex(TemplateError, 'physical template_type'):
            group_measured_slots(self.observations(), {})
        frame, config = self.fixture()
        frame.depth_m[46:175, 168:233] = 0
        with self.assertRaisesRegex(TemplateError, 'Insufficient measured depth'):
            register_template(frame, config)


if __name__ == '__main__':
    unittest.main()
