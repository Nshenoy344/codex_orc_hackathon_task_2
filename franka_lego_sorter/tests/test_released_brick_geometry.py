import unittest
import numpy as np
from lego_sorter.released_brick_geometry import static_bounds,review_hand

class ReleasedBrickGeometryTests(unittest.TestCase):
    def fixture(self):
        anchor=np.diag([1.,-1.,-1.,1.]);anchor[2,3]=.05
        release={'T_base_tcp_before':anchor.tolist()};target={'estimated_bottom_footprint_tcp_m':[[-.016,-.008,.016],[.016,-.008,.016],[.016,.008,.016],[-.016,.008,.016]],'bottom_plane_tcp':{'normal':[0,0,-1]}}
        return static_bounds(release,target)
    def test20mm_retreat_passes_without_large_initial_clearance(self):
        a,lo,hi=self.fixture();up=a.copy();up[2,3]+=.3
        self.assertTrue(review_hand([a,up],.020,a,lo,hi)['static_brick_hand_clear'])
    def test_closed_holding_width_would_intersect_static_released_proxy(self):
        a,lo,hi=self.fixture();self.assertFalse(review_hand([a],.01535,a,lo,hi)['static_brick_hand_clear'])
    def test_settling_union_and_margin_are_retained(self):
        _,lo,hi=self.fixture();self.assertAlmostEqual(lo[2],.0036);self.assertAlmostEqual(hi[2],.020)
