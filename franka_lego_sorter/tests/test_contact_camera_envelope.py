import unittest
import numpy as np
from lego_sorter.contact_camera_envelope import review_contact_camera_envelope,verify_vendor_containment

class ContactCameraEnvelopeTests(unittest.TestCase):
    def pose(self,z=.1):
        t=np.diag([1.,-1.,-1.,1.]);t[2,3]=z;return t
    def run_review(self,cloud,z=.1):
        return review_contact_camera_envelope([self.pose(z)],np.eye(4),cloud,[0,0,1],[0,0,0])
    def test_free_envelope(self):self.assertTrue(self.run_review([[1.,1.,1.]])['passed'])
    def test_offcenter_rgb_housing_width_is_covered(self):
        r=self.run_review([[.09,0,.1]])
        self.assertEqual(r['maximum_camera_observed_intersections'],1);self.assertFalse(r['passed'])
    def test_camera_front_table_intersection_refused(self):
        r=self.run_review([[1.,1.,1.]],z=.005)
        self.assertLess(r['minimum_camera_table_clearance_m'],0);self.assertFalse(r['passed'])
    def test_empty_cloud_cannot_certify_clearance(self):
        with self.assertRaises(ValueError):self.run_review(np.empty((0,3)))

    def test_vendor_collision_front_is_enclosed_after_color_offset(self):
        x=np.eye(4);x[0,3]=.015
        result=verify_vendor_containment(x)
        self.assertTrue(result['assumed_housing_encloses_vendor_box'])
        self.assertGreater(result['vendor_collision_corner_max_color_m'][2],.010)
