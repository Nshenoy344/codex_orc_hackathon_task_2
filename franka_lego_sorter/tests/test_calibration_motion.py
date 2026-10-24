from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import cv2
import numpy as np
from lego_sorter.camera import Frame
from lego_sorter.calibration_motion import rotation_target, scene_guard, slow_plan, validate_rotation_plan, retreat_target, retreat_scene_guard


def pose(rv=(0,0,0), xyz=(.0,.0,.35)):
    p=np.eye(4);p[:3,:3]=cv2.Rodrigues(np.array(rv,float))[0];p[:3,3]=xyz;return p


def data_for_plan(z=.35, lever=0):
    start=pose(xyz=(0,0,z+lever));target=rotation_target(start,start,'x',3)
    return dict(base_frame='base',tcp_frame='tcp',fraction=1.,joint_names=['joint1'],
        T_base_tcp_start=start.tolist(),T_base_tcp_target=target.tolist(),
        calibration_rotation=dict(T_base_tcp_reference=start.tolist(),max_step_deg=3.1),
        points=[dict(positions=[q],velocities=[0.],accelerations=[0.],time_sec=t)
                for q,t in [(0.,0.),(np.radians(3),8.)]])


def urdf(z=.35,lever=0,limit=1.):
    return f'''<robot name="test"><link name="base"/><link name="joint_link"/><link name="tcp"/>
    <joint name="joint1" type="revolute"><parent link="base"/><child link="joint_link"/>
    <origin xyz="0 0 {z}"/><axis xyz="1 0 0"/><limit lower="-{limit}" upper="{limit}" velocity=".5" effort="10"/></joint>
    <joint name="tip" type="fixed"><parent link="joint_link"/><child link="tcp"/><origin xyz="0 0 {lever}"/></joint></robot>'''


class CalibrationMotionTests(unittest.TestCase):
    def test_reference_absolute_orientation_and_fixed_tip(self):
        reference=pose()
        start=pose(rv=(np.radians(3),0,0))
        target=rotation_target(start,reference,'x',6)
        np.testing.assert_allclose(target[:3,3],start[:3,3])
        np.testing.assert_allclose(target[:3,:3],pose(rv=(np.radians(6),0,0))[:3,:3])

    def test_retreat_preserves_center_and_reduces_to_observed_orientation(self):
        reference=pose()
        start=pose(rv=(np.radians(15),0,0),xyz=(.001,0,.35))
        accepted=pose(rv=(np.radians(12),0,0))
        target=retreat_target(start,reference,accepted)
        np.testing.assert_allclose(target[:3,3],start[:3,3])
        np.testing.assert_allclose(target[:3,:3],accepted[:3,:3])
        with self.assertRaisesRegex(ValueError,'exceeds'):
            retreat_target(start,reference,pose(rv=(np.radians(11),0,0)))
        with self.assertRaisesRegex(ValueError,'reduce'):
            retreat_target(accepted,reference,start)

    def test_explicit_reviewed_retreat_bound_handles_measured_tracking_error(self):
        reference=pose()
        start=pose(rv=(np.radians(12.4),0,0))
        accepted=pose(rv=(np.radians(9),0,0))
        with self.assertRaisesRegex(ValueError,'exceeds'):
            retreat_target(start,reference,accepted)
        target=retreat_target(start,reference,accepted,max_step_deg=5.1)
        np.testing.assert_allclose(target[:3,:3],accepted[:3,:3])
        with self.assertRaisesRegex(ValueError,'5.1'):
            retreat_target(start,reference,accepted,max_step_deg=5.2)
        with self.assertRaisesRegex(ValueError,'reduce'):
            retreat_target(accepted,reference,start,max_step_deg=5.1)

    def test_retreat_navigation_does_not_relax_calibration_admission(self):
        frame=Frame(np.zeros((40,40,3),np.uint8),np.full((40,40),.3),
                    dict(fx=100.,fy=100.,ppx=20.,ppy=20.),0.,dict(serial='unit'))
        def observe(f,config,max_rms_px):
            rms=.63 if max_rms_px==.7 else .3
            return dict(T_color_board=np.eye(4).tolist(),metrics=dict(reprojection_rms_px=rms))
        with patch('lego_sorter.charuco_handeye.observe_board',side_effect=observe) as checked:
            guard=retreat_scene_guard(frame,frame,frame,{})
        self.assertEqual([v.kwargs['max_rms_px']for v in checked.call_args_list],[.6,.6,.7])
        self.assertFalse(guard['current_observation_added_to_calibration'])
        self.assertFalse(guard['current_would_pass_calibration_rms'])
        self.assertEqual(guard['calibration_admission_max_rms_px'],.6)

    def test_explicit_candidate_budget_is_recorded_and_capped(self):
        frame=Frame(np.zeros((40,40,3),np.uint8),np.full((40,40),.3),
                    dict(fx=600.,fy=600.,ppx=20.,ppy=20.),0.,dict(serial='unit'))
        target=np.eye(4);target[2,3]=.37
        def observe(f,config,max_rms_px):
            if max_rms_px<.63:raise ValueError('Reprojection RMS exceeds selectedbudget')
            return dict(T_color_board=target.tolist(),metrics=dict(reprojection_rms_px=.63,corner_count=22))
        with patch('lego_sorter.charuco_handeye.observe_board',side_effect=observe):
            with self.assertRaisesRegex(ValueError,'selectedbudget'):
                scene_guard(frame,frame,board_config={})
            guard=scene_guard(frame,frame,board_config={},max_board_rms_px=.8)
            self.assertEqual(guard['registration']['admission_max_rms_px'],.8)
            self.assertAlmostEqual(guard['registration']['current_image_plane_rms_equivalent_m'],.63/600*.37)
            with self.assertRaisesRegex(ValueError,'0.8'):
                scene_guard(frame,frame,board_config={},max_board_rms_px=.81)

    def test_explicit_center_drift_budget_does_not_command_translation(self):
        reference=pose()
        start=pose(xyz=(.0027,.0008,.3449))
        with self.assertRaisesRegex(ValueError,'acquisition envelope'):
            rotation_target(start,reference,'x',3)
        target=rotation_target(start,reference,'x',3,max_center_drift_m=.010)
        np.testing.assert_array_equal(target[:3,3],start[:3,3])
        with self.assertRaisesRegex(ValueError,'0.010'):
            rotation_target(start,reference,'x',3,max_center_drift_m=.011)

    def test_low_height_and_large_step_refused(self):
        with self.assertRaisesRegex(ValueError,'Raise'):
            rotation_target(pose(xyz=(0,0,.30)),pose(),'x',3)
        with self.assertRaisesRegex(ValueError,'step'):
            rotation_target(pose(),pose(),'x',10)
        with self.assertRaisesRegex(ValueError,'15'):
            rotation_target(pose(),pose(),'x',16)
        with self.assertRaisesRegex(ValueError,'restricted'):
            rotation_target(pose(),pose(),'z',3)

    def test_slowing_preserves_curve_derivatives(self):
        points=[SimpleNamespace(time_from_start=SimpleNamespace(sec=t,nanosec=0),
                                velocities=[.04],accelerations=[.02]) for t in (0,2)]
        plan=SimpleNamespace(noop=False,trajectory=SimpleNamespace(joint_trajectory=SimpleNamespace(points=points)))
        slow_plan(plan)
        self.assertEqual(points[-1].time_from_start.sec,8)
        self.assertAlmostEqual(points[0].velocities[0],.01)
        self.assertAlmostEqual(points[0].accelerations[0],.00125)

    def test_exact_dense_rotation_validation(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'robot.urdf';p.write_text(urdf())
            report=validate_rotation_plan(data_for_plan(),p)
            self.assertTrue(report['passed_numeric_checks'])
            self.assertGreater(report['sample_count'],800)
            self.assertAlmostEqual(report['max_translation_drift_m'],0.)
            self.assertAlmostEqual(report['maximum_reference_angle_deg'],3.)
            self.assertFalse(report['camera_collision_modeled'])

    def test_dense_translation_drift_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'robot.urdf';p.write_text(urdf(lever=.02))
            report=validate_rotation_plan(data_for_plan(lever=.02),p)
            self.assertFalse(report['passed_numeric_checks'])
            self.assertGreater(report['max_translation_drift_m'],.0005)

    def test_joint_limit_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'robot.urdf';p.write_text(urdf(limit=.04))
            report=validate_rotation_plan(data_for_plan(),p)
            self.assertFalse(report['passed_numeric_checks'])
            self.assertFalse(report['joint_limits']['joint1']['passed'])

    def test_observed_depth_and_real_overlap_required(self):
        frame=Frame(np.zeros((30,30,3),np.uint8),np.full((30,30),.3),
                    dict(fx=100.,fy=100.,ppx=15.,ppy=15.),0.,dict(serial='unit'))
        registered=SimpleNamespace(metrics=dict(inlier_count=50,inlier_fraction=.9))
        with patch('lego_sorter.calibration_motion.register_rgbd',return_value=registered) as register:
            result=scene_guard(frame,frame)
            self.assertAlmostEqual(result['nearest_depth_percentile2_m'],.3)
            register.assert_called_once()
        frame.depth_m[:]=.1
        with self.assertRaisesRegex(ValueError,'nearest'):
            scene_guard(frame,frame)


if __name__=='__main__':unittest.main()
