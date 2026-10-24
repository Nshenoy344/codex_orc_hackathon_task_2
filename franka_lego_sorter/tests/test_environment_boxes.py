import unittest
import numpy as np
from lego_sorter.environment_boxes import intersections

class EnvironmentBoxTests(unittest.TestCase):
    def setUp(self):self.vertices=np.array([[0.,0.,0.],[1,0,0],[0,1,0],[0,0,1]])
    def test_true_hull_box_intersection(self):
        self.assertEqual(intersections(self.vertices,[('obstacle',np.array([.1,.1,.1]),np.array([.2,.2,.2]))]),['obstacle'])
    def test_aabb_overlap_alone_does_not_claim_mesh_collision(self):
        self.assertEqual(intersections(self.vertices,[('clear',np.array([.8,.8,.8]),np.array([.9,.9,.9]))]),[])
    def test_broad_phase_separation(self):
        self.assertEqual(intersections(self.vertices,[('far',np.array([2.,2.,2.]),np.array([3.,3.,3.]))]),[])
