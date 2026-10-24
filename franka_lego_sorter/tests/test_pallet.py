import unittest
import numpy as np
from lego_sorter.camera import Frame
from lego_sorter.perception import ColorModel
from lego_sorter.pallet import inspect_pallet, jaw_clearance, fit_brick_body_top


class PalletTests(unittest.TestCase):
    def scene(self):
        image = np.full((240, 320, 3), 180, np.uint8)
        depth = np.full((240, 320), .40, np.float32)
        image[105:135, 130:190] = (20, 30, 180)
        depth[105:135, 130:190] = .32
        return Frame(image, depth, dict(fx=600., fy=600., ppx=160., ppy=120.), 0.)

    def model(self):
        model = ColorModel()
        for label, color in [('taught_red', (20, 30, 180)), ('taught_tan', (100, 160, 190))]:
            model.teach(label, np.full((30, 30, 3), color, np.uint8), [2, 2, 26, 26])
        return model

    def test_isolated_geometry_and_taught_color(self):
        inspection = inspect_pallet(self.scene(), {'segmentation_clusters': 2, 'require_body_top_reference':False}, self.model())
        self.assertEqual(len(inspection.candidates), 1)
        candidate = inspection.candidates[0]
        self.assertEqual(candidate.color_id, 'taught_red')
        self.assertTrue(candidate.graspable, candidate.rejection_reasons)
        self.assertGreater(candidate.clearance_fraction, .9)
        self.assertLess(candidate.surface_plane_rms_m, 1e-6)
        np.testing.assert_allclose(candidate.surface_normal_camera, [0, 0, -1], atol=1e-6)
        np.testing.assert_allclose(candidate.point_camera_m, [0, 0, .32], atol=.001)

    def test_no_model_does_not_invent_physical_shade(self):
        inspection = inspect_pallet(self.scene(), {'segmentation_clusters': 2, 'require_body_top_reference':False})
        self.assertIsNone(inspection.candidates[0].color_id)
        self.assertFalse(inspection.candidates[0].graspable)
        self.assertIn('physical_shade_unverified', inspection.candidates[0].rejection_reasons)

    def test_neighbor_blocks_one_jaw(self):
        frame = self.scene()
        frame.depth_m[85:105, 130:190] = .32
        fraction, reasons = jaw_clearance(frame, [160, 120], [1, 0], 60, 30, .32, {})
        self.assertLess(fraction, .2)
        self.assertIn('neighbor_or_unknown_depth_in_jaw_strip', reasons)

    def test_end_grasp_selected_when_width_grasp_blocked(self):
        frame = self.scene()
        frame.depth_m[85:105, 130:190] = .32
        inspection = inspect_pallet(frame, {'segmentation_clusters': 2, 'require_body_top_reference':False}, self.model())
        candidate = inspection.candidates[0]
        self.assertEqual(candidate.grasp_orientation, 'across_length')
        self.assertTrue(candidate.graspable, candidate.rejection_reasons)
        self.assertAlmostEqual(candidate.grasp_width_m, .0315, places=3)
        self.assertGreater(abs(candidate.grasp_closing_axis_camera[0]), .99)

    def test_missing_depth_is_not_free_space(self):
        frame = self.scene()
        frame.depth_m[85:105, 130:190] = 0
        fraction, reasons = jaw_clearance(frame, [160, 120], [1, 0], 60, 30, .32, {})
        self.assertLess(fraction, .2)
        self.assertTrue(reasons)

    def test_jaw_opening_margin_expands_the_required_depth_strip(self):
        frame = self.scene()
        # Closed10mm-thick jaw ends before these pixels; the extra2mm opening sweep hits them.
        frame.depth_m[156:161,130:190] = .32
        closed, _ = jaw_clearance(frame,[160,120],[1,0],60,30,.32,
                                  {'finger_clearance_m':.010,'jaw_opening_margin_m':0.})
        swept, reasons = jaw_clearance(frame,[160,120],[1,0],60,30,.32,
                                      {'finger_clearance_m':.010,'jaw_opening_margin_m':.004})
        self.assertGreater(closed,.99)
        self.assertLess(swept,.9)
        self.assertIn('neighbor_or_unknown_depth_in_jaw_strip',reasons)

    def test_nominal_width_mode_retains_measured_geometry_and_source(self):
        frame = self.scene()
        inspection = inspect_pallet(frame, {'segmentation_clusters':2, 'require_body_top_reference':False,
            'grasp_width_source':'configured_2x4_nominal', 'jaw_opening_margin_m':.001}, self.model())
        brick = inspection.candidates[0]
        self.assertEqual(brick.grasp_width_source,'configured_2x4_nominal')
        self.assertAlmostEqual(brick.grasp_width_m,.0158)
        self.assertAlmostEqual(brick.measured_grasp_width_m,brick.width_m)
        self.assertNotEqual(brick.measured_grasp_width_m,brick.grasp_width_m)
        option = brick.clearance_options[brick.grasp_orientation]
        self.assertAlmostEqual(option['body_width_m'],.0158)
        self.assertAlmostEqual(option['measured_width_m'],brick.width_m)

    def test_nominal_long_grasp_width_is_31_8_mm(self):
        frame = self.scene()
        frame.depth_m[85:105,130:190] = .32
        brick = inspect_pallet(frame, {'segmentation_clusters':2, 'require_body_top_reference':False,
            'grasp_width_source':'configured_2x4_nominal'}, self.model()).candidates[0]
        self.assertEqual(brick.grasp_orientation,'across_length')
        self.assertAlmostEqual(brick.grasp_width_m,.0318)
        self.assertAlmostEqual(brick.measured_grasp_width_m,brick.length_m)

    def studded_scene(self):
        frame=self.scene()
        scale=600./.32
        for along in (-.012,-.004,.004,.012):
            for across in (-.004,.004):
                center=(round(159.5+along*scale),round(119.5+across*scale))
                import cv2
                cv2.circle(frame.depth_m,center,round(.0024*scale),.32-.0018,-1)
        return frame

    def test_measured_body_strip_and_stud_relief_define_body_reference(self):
        frame=self.studded_scene()
        center,normal,rms,evidence=fit_brick_body_top(frame,[159.5,119.5],[1.,0.],59.,29.,.32)
        self.assertTrue(evidence['verified'],evidence)
        self.assertEqual(evidence['resolved_stud_count'],8)
        self.assertFalse(evidence['nominal_height_subtracted'])
        self.assertAlmostEqual(float(center[2]),.32,places=6)
        for relief in evidence['measured_stud_relief_m']:
            self.assertAlmostEqual(relief,.0018,places=6)

    def test_flat_depth_does_not_invent_body_reference(self):
        frame=self.scene()
        _,_,_,evidence=fit_brick_body_top(frame,[159.5,119.5],[1.,0.],59.,29.,.32)
        self.assertFalse(evidence['verified'])
        self.assertEqual(evidence['resolved_stud_count'],0)
        inspection=inspect_pallet(frame,{'segmentation_clusters':2},self.model())
        self.assertEqual(len(inspection.candidates),1)
        candidate=inspection.candidates[0]
        self.assertFalse(candidate.graspable)
        self.assertIn('body_top_plane_unverified',candidate.rejection_reasons)
        self.assertEqual(candidate.surface_reference,'unverified_face')
        self.assertAlmostEqual(candidate.depth_m,.32,places=6) # No invented1.8mm subtraction.

    def test_verified_body_reference_reaches_candidate_and_keeps_body_depth(self):
        inspection=inspect_pallet(self.studded_scene(),{'segmentation_clusters':2},self.model())
        candidate=inspection.candidates[0]
        self.assertEqual(candidate.surface_reference,'measured_inter_stud_body_top',candidate.body_top_diagnostics)
        self.assertNotIn('body_top_plane_unverified',candidate.rejection_reasons)
        self.assertTrue(candidate.graspable,candidate.rejection_reasons)
        self.assertAlmostEqual(candidate.point_camera_m[2],.32,places=6)

    def test_body_strip_missing_depth_stops_reference_estimation(self):
        frame=self.studded_scene()
        frame.depth_m[118:122,135:185]=0
        with self.assertRaisesRegex(ValueError,'Insufficient reliable depth'):
            fit_brick_body_top(frame,[159.5,119.5],[1.,0.],59.,29.,.32)

    def test_body_support_pixel_thresholds_scale_with_physical_resolution(self):
        import cv2
        low=self.studded_scene()
        _,_,_,low_quality=fit_brick_body_top(low,[159.5,119.5],[1.,0.],59.,29.,.32)
        high=Frame(cv2.resize(low.color_bgr,None,fx=2,fy=2,interpolation=cv2.INTER_NEAREST),
                   cv2.resize(low.depth_m,None,fx=2,fy=2,interpolation=cv2.INTER_NEAREST),
                   dict(fx=1200.,fy=1200.,ppx=320.,ppy=240.),0.)
        _,_,_,high_quality=fit_brick_body_top(high,[319.5,239.5],[1.,0.],118.,58.,.32)
        self.assertTrue(high_quality['verified'],high_quality)
        self.assertGreater(high_quality['minimum_strip_pixels'],low_quality['minimum_strip_pixels'])
        self.assertGreater(high_quality['minimum_stud_pixels'],low_quality['minimum_stud_pixels'])
        self.assertEqual(high_quality['support_minimum_physical_areas_m2'],low_quality['support_minimum_physical_areas_m2'])

    def test_narrow_quantized_strip_does_not_invent_opposite_stud_row_heights(self):
        frame=self.studded_scene()
        # The stripe has a quantization-scale cross-strip slope unsupported by
        # the rest of the face. Fitting a free normal to these4rows is ill-conditioned.
        for row,delta in [(118,.0008),(119,.0003),(120,-.0003),(121,-.0008)]:
            frame.depth_m[row,136:184]=.32+delta
        _,normal,_,quality=fit_brick_body_top(frame,[159.5,119.5],[1.,0.],59.,29.,.32)
        self.assertTrue(quality['verified'],quality)
        self.assertTrue(quality['body_strip_fits_offset_only'])
        self.assertEqual(quality['plane_normal_source'],'independently_measured_wide_face')
        self.assertLess(np.linalg.norm(normal[:2]),.1)
        self.assertTrue(all(value>0 for value in quality['measured_stud_relief_m']))

    def test_oblique_same_height_neighbor_is_not_free_from_optical_depth(self):
        frame=self.scene()
        angle=np.radians(15.) # Within the existing18deg near-downward assumption.
        normal=np.array([0.,np.sin(angle),-np.cos(angle)])
        rows,cols=np.indices(frame.depth_m.shape)
        rays_y=(rows-frame.intrinsics['ppy'])/frame.intrinsics['fy']
        plane_depth=(-.32*np.cos(angle))/(rays_y*np.sin(angle)-np.cos(angle))
        frame.depth_m[151:181,140:181]=plane_depth[151:181,140:181]
        legacy,_=jaw_clearance(frame,[160,120],[1,0],30,60,.32,
                              {'grasp_depth_below_surface_m':.0015,'clearance_depth_margin_m':.002})
        plane_aware,reasons=jaw_clearance(frame,[160,120],[1,0],30,60,.32,
                              {'grasp_depth_below_surface_m':.0015,'clearance_depth_margin_m':.002},
                              normal,[0.,0.,.32])
        self.assertGreater(legacy,.95) # ConstantZ would falsely approve the same-level neighbor.
        self.assertLess(plane_aware,.05)
        self.assertIn('neighbor_or_unknown_depth_in_jaw_strip',reasons)

    def test_oblique_really_lower_surface_has_measured_plane_clearance(self):
        frame=self.scene()
        angle=np.radians(15.)
        normal=np.array([0.,np.sin(angle),-np.cos(angle)])
        rows,cols=np.indices(frame.depth_m.shape)
        denominators=(rows-frame.intrinsics['ppy'])/frame.intrinsics['fy']*normal[1]+normal[2]
        frame.depth_m[:]=(-.32*np.cos(angle)-.012)/denominators
        fraction,reasons=jaw_clearance(frame,[160,120],[1,0],30,60,.32,
                            {'grasp_depth_below_surface_m':.0015,'clearance_depth_margin_m':.002},
                            normal,[0.,0.,.32])
        self.assertGreater(fraction,.99)
        self.assertEqual(reasons,[])

    def test_long_same_shade_region_is_ambiguous(self):
        frame = self.scene()
        frame.color_bgr[105:135, 190:250] = (20, 30, 180)
        frame.depth_m[105:135, 190:250] = .32
        inspection = inspect_pallet(frame, {'segmentation_clusters': 2, 'require_body_top_reference':False}, self.model())
        self.assertEqual(len(inspection.candidates), 2)
        for candidate in inspection.candidates:
            self.assertTrue(candidate.split_region)
            self.assertFalse(candidate.graspable)
            self.assertIn('same_color_region_split_boundary_unverified', candidate.rejection_reasons)


if __name__ == '__main__':
    unittest.main()
