import unittest
import numpy as np
from lego_sorter.camera import Frame
from lego_sorter.perception import ColorModel,lab_feature
from lego_sorter.placement_verification import verify_placement
class PlacementVerificationTests(unittest.TestCase):
 def setup_frames(self):
  b=np.full((160,160,3),(0,90,180),np.uint8);a=np.full((160,160,3),(25,20,60),np.uint8);mask=np.ones((160,160),np.uint8);model=ColorModel({'6487946':[lab_feature(b,mask).tolist()],'6117418':[lab_feature(a,mask).tolist()]});intr=dict(fx=500.,fy=500.,ppx=80.,ppy=80.);before=Frame(b,np.full((160,160),.5,np.float32),intr,0.,{});after=Frame(a,np.full((160,160),.4904,np.float32),intr,1.,{});slot=dict(point_camera_m=[0,0,.5],surface_normal_camera=[0,0,-1],box_px=[[60,40],[100,40],[100,120],[60,120]]);return before,after,slot,model
 def test_measures_observed_increment_without_committing_pitch_or_counts(self):
  b,a,s,m=self.setup_frames();r,_=verify_placement(b,a,np.eye(4),np.eye(4),s,m);self.assertEqual(r['status'],'consistent_with_held_brick_placement');self.assertAlmostEqual(r['measured_visible_top_increment_m'],.0096,places=6);self.assertFalse(r['production_counts_committed']);self.assertFalse(r['measured_layer_pitch_verified'])
 def test_wrong_or_uncertain_color_is_unknown(self):
  b,a,s,m=self.setup_frames();a.color_bgr[:]=b.color_bgr;r,_=verify_placement(b,a,np.eye(4),np.eye(4),s,m);self.assertEqual(r['status'],'UNKNOWN')
 def test_missing_depth_refuses_evidence(self):
  b,a,s,m=self.setup_frames();a.depth_m[:]=0
  with self.assertRaisesRegex(ValueError,'Insufficient actual depth'):verify_placement(b,a,np.eye(4),np.eye(4),s,m)
