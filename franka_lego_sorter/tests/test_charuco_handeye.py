import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import cv2
import numpy as np
from lego_sorter.camera import Frame
from lego_sorter.charuco_observation import _core
from lego_sorter.charuco_handeye import observe_board,calibrate_charuco,_metric_rigid_pose
from lego_sorter.scene_handeye import CalibrationRejected


def frame_and_config():
    config=dict(dictionary='DICT_4X4_1000',squares_x=7,squares_y=5,square_length_m=.036,
                marker_length_m=.0252,legacy_pattern=False)
    board,_=_core().make_board(config)
    image=board.generateImage((630,450)) if hasattr(board,'generateImage') else board.draw((630,450))
    image=cv2.copyMakeBorder(image,100,100,100,100,cv2.BORDER_CONSTANT,value=255)
    h,w=image.shape
    return Frame(cv2.cvtColor(image,cv2.COLOR_GRAY2BGR),np.full((h,w),.4,np.float32),
                 dict(fx=1000.,fy=1000.,ppx=w/2,ppy=h/2,width=w,height=h,coeffs=[0]*5),
                 0.,dict(serial='unit',exposure='fixed')),config


def pose(rv,xyz):
    p=np.eye(4);p[:3,:3]=cv2.Rodrigues(np.asarray(rv,float))[0];p[:3,3]=xyz;return p


def save(path,frame,robot,extrinsic=None):
    data=dict(color_bgr=frame.color_bgr,depth_m=frame.depth_m,
              intrinsics_json=json.dumps(frame.intrinsics),metadata_json=json.dumps(frame.metadata),
              captured_monotonic=0.,T_base_tcp=robot,robot_drift_m=0.,robot_drift_rad=0.)
    if extrinsic is not None:data['T_color_depth']=extrinsic
    np.savez_compressed(path,**data)


class CharucoHandeyeTests(unittest.TestCase):
    def test_known_board_pnp_and_depth_plane(self):
        frame,cfg=frame_and_config();result=observe_board(frame,cfg)
        self.assertEqual(result['metrics']['corner_count'],24)
        self.assertLess(result['metrics']['reprojection_rms_px'],.6)
        self.assertTrue(result['metrics']['depth_consistency']['passed'])
        self.assertLess(result['metrics']['depth_consistency']['median_depth_residual_m'],.001)
        self.assertFalse(result['physical_metric_scale_verified'])

    def test_rgbd_plane_estimator_retains_pnp_and_fixed_scale(self):
        frame,cfg=frame_and_config();result=observe_board(frame,cfg,estimator='rgbd_plane')
        self.assertEqual(result['metrics']['pose_estimator'],'rgbd_plane')
        self.assertIn('T_color_board_pnp',result)
        self.assertAlmostEqual(result['T_color_board'][2][3],.4,places=3)
        self.assertLess(result['metrics']['rgbd_metric_fit']['metric_rms_m'],.0015)
        self.assertEqual(result['fixed_board_config'],cfg)

    def test_rgbd_plane_cannot_bypass_depth_guard(self):
        frame,cfg=frame_and_config()
        with self.assertRaisesRegex(ValueError,'requires qualified'):
            observe_board(frame,cfg,require_depth_consistency=False,estimator='rgbd_plane')
        frame.depth_m[:]=0.
        with self.assertRaises(ValueError):
            observe_board(frame,cfg,estimator='rgbd_plane')

    def test_metric_planar_kabsch_known_pose(self):
        source=np.array([[x*.03,y*.03,0.]for x in range(6)for y in range(4)])
        expected=pose([.12,-.15,.3],[.04,-.02,.4])
        observed=source@expected[:3,:3].T+expected[:3,3]
        fitted,metrics=_metric_rigid_pose(source,observed)
        np.testing.assert_allclose(fitted,expected,atol=1e-10)
        self.assertLess(metrics['metric_max_m'],1e-10)

    def test_metric_corner_identity_mismatch_rejected(self):
        source=np.array([[x*.03,y*.03,0.]for x in range(6)for y in range(4)])
        observed=source+np.array([0,0,.4]);observed[[0,7]]=observed[[7,0]]
        with self.assertRaisesRegex(ValueError,'metric corner fit failed'):
            _metric_rigid_pose(source,observed)

    def test_partial_known_board_does_not_require_all_markers(self):
        frame,cfg=frame_and_config()
        frame.color_bgr=frame.color_bgr[:400].copy();frame.depth_m=frame.depth_m[:400].copy()
        result=observe_board(frame,cfg)
        self.assertGreaterEqual(result['metrics']['corner_count'],12)
        self.assertLess(result['metrics']['corner_count'],24)

    def test_wrong_fixed_metric_scale_rejected_by_depth(self):
        frame,cfg=frame_and_config();cfg['square_length_m']=.04;cfg['marker_length_m']=.028
        with self.assertRaisesRegex(ValueError,'consistency failed'):
            observe_board(frame,cfg)

    def test_missing_depth_rejected_for_estimated_scale(self):
        frame,cfg=frame_and_config();frame.depth_m[:]=0.
        with self.assertRaisesRegex(ValueError,'depth'):
            observe_board(frame,cfg)

    def test_insufficient_batch_retains_target_observation_no_calibration(self):
        frame,cfg=frame_and_config()
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'one.npz';save(path,frame,np.eye(4))
            with self.assertRaises(CalibrationRejected) as error:
                calibrate_charuco([path],cfg)
        diagnostics=error.exception.diagnostics
        self.assertFalse(diagnostics['accepted'])
        self.assertEqual(len(diagnostics['observations']),1)
        self.assertIn('T_color_board',diagnostics['observations'][0])
        self.assertNotIn('T_tcp_color',diagnostics)

    def test_batch_composes_factory_depth_and_preserves_scale_provenance(self):
        frame,cfg=frame_and_config();rng=np.random.default_rng(93)
        x=pose([.2,-.12,.3],[.04,-.03,.1]);fixed=pose([0.,0.,0.],[.4,0.,.1])
        robots=[pose(rng.normal(0,.4,3),rng.uniform([.25,-.15,.3],[.5,.15,.55])) for _ in range(18)]
        targets=[np.linalg.inv(x)@np.linalg.inv(g)@fixed for g in robots]
        extrinsic=pose([0.,.01,0.],[.015,0.,0.])
        observations=[dict(T_color_board=t.tolist(),metrics=dict(corner_count=24,reprojection_rms_px=.1)) for t in targets]
        with tempfile.TemporaryDirectory() as d:
            paths=[]
            for i,g in enumerate(robots):
                p=Path(d)/f'{i}.npz';save(p,frame,g,extrinsic);paths.append(p)
            with patch('lego_sorter.charuco_handeye.observe_board',side_effect=observations):
                result=calibrate_charuco(paths,cfg)
        np.testing.assert_allclose(result['T_tcp_color'],x,atol=1e-8)
        np.testing.assert_allclose(result['T_tcp_depth'],x@extrinsic,atol=1e-8)
        np.testing.assert_allclose(np.array(result['T_depth_tcp'])@np.array(result['T_tcp_depth']),np.eye(4),atol=1e-8)
        self.assertEqual(result['parameters']['tcp_frame'],'fr3_hand_tcp')
        self.assertEqual(result['scale_source'],'aligned_depth_estimate')
        self.assertFalse(result['physical_scale_verified'])
        self.assertFalse(result['physical_commissioning_verified'])
        self.assertTrue(result['diagnostics']['accepted'])

    def test_factory_extrinsic_change_rejected(self):
        frame,cfg=frame_and_config();a=np.eye(4);b=np.eye(4);b[0,3]=.001
        with tempfile.TemporaryDirectory() as d:
            first=Path(d)/'a.npz';second=Path(d)/'b.npz'
            save(first,frame,np.eye(4),a);save(second,frame,np.eye(4),b)
            with self.assertRaisesRegex(CalibrationRejected,'extrinsics changed'):
                calibrate_charuco([first,second],cfg)


if __name__=='__main__':unittest.main()
