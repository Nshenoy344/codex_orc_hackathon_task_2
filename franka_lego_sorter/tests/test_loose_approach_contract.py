import importlib.util,json,sys,tempfile,unittest
from pathlib import Path
TOOLS=Path(__file__).parents[1]/'tools';sys.path.insert(0,str(TOOLS))
spec=importlib.util.spec_from_file_location('loose_approach_review',TOOLS/'validate_first_loose_place_approach.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
class LooseApproachContractTests(unittest.TestCase):
 def test_other_stage_cannot_enter_loose_approach_review(self):
  with tempfile.TemporaryDirectory() as root:
   p=Path(root)/'plan.json';p.write_text(json.dumps(dict(isolated_diagnostic_stage=dict(stage='release',gripper_action_authorized=False))))
   with self.assertRaisesRegex(ValueError,'Only no-release'):mod.review(p)
 def test_finger_authorization_rejected_even_for_exact_stage(self):
  with tempfile.TemporaryDirectory() as root:
   p=Path(root)/'plan.json';p.write_text(json.dumps(dict(isolated_diagnostic_stage=dict(stage='first_loose_place_approach',gripper_action_authorized=True))))
   with self.assertRaisesRegex(ValueError,'Only no-release'):mod.review(p)
