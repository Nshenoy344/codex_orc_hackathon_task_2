import importlib.util,json,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).parents[1];spec=importlib.util.spec_from_file_location('release_once',ROOT/'tools/release_first_loose_once.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
class ReleaseFreshChecks(unittest.TestCase):
 def setUp(self):
  self.target=json.loads((ROOT/'data/commissioning/first_loose_placement_target.json').read_text());self.actual=np.array(self.target['T_base_tcp_release'])
 def test_actual_target_retains_measured_slot_clearance(self):
  checks=mod.check_pose_and_hold(self.actual,.01535,self.target);self.assertGreater(checks['minimum_bottom_clearance_above_slot_observed_max_m'],.0028)
 def test_endpoint_drift_prevents_opening(self):
  self.actual[0,3]+=.0012
  with self.assertRaisesRegex(ValueError,'endpoint'):mod.check_pose_and_hold(self.actual,.01535,self.target)
 def test_changed_heldwidth_prevents_opening(self):
  with self.assertRaisesRegex(ValueError,'heldbody'):mod.check_pose_and_hold(self.actual,.016,self.target)
 def test_nan_actual_pose_refused(self):
  self.actual[0,3]=np.nan
  with self.assertRaisesRegex(ValueError,'Invalid'):mod.check_pose_and_hold(self.actual,.01535,self.target)
 def test_conditional_bottom_clearance_is_independently_checked(self):
  self.target['support_plane_base']['offset_p99_m']+=.002
  with self.assertRaisesRegex(ValueError,'heldbottom'):mod.check_pose_and_hold(self.actual,.01535,self.target)
