import unittest
import cv2
import numpy as np
from lego_sorter.camera import Frame
from lego_sorter.charuco_observation import inspect_charuco,scan_dictionaries,_core


def board_frame():
    config=dict(dictionary='DICT_4X4_1000',squares_x=7,squares_y=5,
                square_length_m=.036,marker_length_m=.0252,legacy_pattern=False)
    board,_=_core().make_board(config)
    image=board.generateImage((630,450)) if hasattr(board,'generateImage') else board.draw((630,450))
    image=cv2.copyMakeBorder(image,100,100,100,100,cv2.BORDER_CONSTANT,value=255)
    bgr=cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
    h,w=image.shape
    frame=Frame(bgr,np.full((h,w),.4,np.float32),
                dict(fx=1000.,fy=1000.,ppx=w/2,ppy=h/2,width=w,height=h,coeffs=[0]*5),0.,dict(serial='synthetic'))
    return frame,config


class CharucoObservationTests(unittest.TestCase):
    def test_complete_board_geometry_and_metric_estimate(self):
        frame,_=board_frame()
        result=inspect_charuco(frame)
        self.assertTrue(result['accepted_observation'])
        self.assertFalse(result['physical_calibration_verified'])
        cfg=result['candidate_board_config']
        self.assertEqual((cfg['squares_x'],cfg['squares_y']),(7,5))
        self.assertEqual(cfg['dictionary'],'DICT_4X4_1000')
        self.assertAlmostEqual(cfg['square_length_m'],.036,places=4)
        self.assertEqual(result['candidates'][0]['corner_count'],24)
        self.assertLess(result['candidates'][0]['reprojection_rms_px'],.5)
        self.assertFalse(result['candidates'][0]['metric_scale_verified_by_physical_measurement'])

    def test_wrong_supplied_metric_scale_flagged(self):
        frame,cfg=board_frame();cfg['square_length_m']=.04;cfg['marker_length_m']=.028
        result=inspect_charuco(frame,cfg)
        self.assertTrue(result['accepted_observation'])
        evidence=result['candidates'][0]['supplied_scale']
        self.assertFalse(evidence['scale_agrees'])
        self.assertAlmostEqual(evidence['depth_estimate_ratio'],.9,places=2)

    def test_missing_depth_rejected(self):
        frame,_=board_frame();frame.depth_m[:]=0.
        result=inspect_charuco(frame)
        self.assertFalse(result['accepted_observation'])
        self.assertNotIn('candidate_board_config',result)

    def test_untextured_image_has_no_assumed_geometry(self):
        frame,_=board_frame();frame.color_bgr[:]=255
        result=inspect_charuco(frame)
        self.assertFalse(result['accepted_observation'])
        self.assertFalse(result['dictionary_scan'])

    def test_clipped_board_not_accepted_as_new_complete_layout(self):
        frame,_=board_frame()
        frame.color_bgr=frame.color_bgr[:400].copy();frame.depth_m=frame.depth_m[:400].copy()
        result=inspect_charuco(frame)
        self.assertFalse(result['accepted_observation'])


if __name__=='__main__':unittest.main()
