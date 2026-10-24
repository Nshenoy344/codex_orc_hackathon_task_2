import importlib.util,unittest
from pathlib import Path
import numpy as np
spec=importlib.util.spec_from_file_location('object_closure',Path(__file__).parents[1]/'tools/grasp_object_once.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
class FreshObjectClosureTests(unittest.TestCase):
 def setUp(self):
  self.goal=np.diag([1.,-1.,-1.,1.]);self.goal[2,3]=.05;self.target=dict(normal_base=[0,0,1],visible_upper_p99_offset_base_m=.05,toe_center_insertion_below_observed_p99_m=.003,minimum_toe_clearance_above_prior_lower_layer_globalmax_m=.017)
 def test_exact_goal_preopen_and_lowerclearance_pass(self):self.assertGreater(mod.fresh_checks(self.goal,.030,self.goal,self.target)['minimum_toe_above_prior_lower_layer_m'],.003)
 def test_width_change_prevents_closure(self):
  with self.assertRaisesRegex(ValueError,'preopen'):mod.fresh_checks(self.goal,.028,self.goal,self.target)
 def test_endpoint_change_prevents_closure(self):
  actual=self.goal.copy();actual[0,3]+=.0011
  with self.assertRaisesRegex(ValueError,'physical goal'):mod.fresh_checks(actual,.030,self.goal,self.target)
 def test_actual_toe_lowerclearance_prevents_closure(self):
  self.target['minimum_toe_clearance_above_prior_lower_layer_globalmax_m']=.007
  with self.assertRaisesRegex(ValueError,'fulltoe'):mod.fresh_checks(self.goal,.030,self.goal,self.target)
